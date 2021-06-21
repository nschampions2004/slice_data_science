library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(skimr)

data_folder <- "data"
plots_folder <- "plots"
predictions_folder <- "predictions"
models_folder <- "models"

theme_set(theme_minimal())


income_data <- readxl::read_xls(file.path(data_folder, "tabn025.xls")) %>% 
  janitor::clean_names() %>% 
  tidyr::separate(col = state, sep = " \\.",
                  into = c("state", "trash")) %>% 
  filter(state != "District") %>% 
  select(state, median_household_income = x2010)

store_training <- read_csv(file.path(data_folder, "train.csv")) %>% 
  left_join(income_data, by = "state") %>% 
  mutate_if(is.character, as.factor) 
store_test <- read_csv(file.path(data_folder, "test.csv")) %>% 
  left_join(income_data, by = "state") %>% 
  mutate_if(is.character, as.factor)

# Alright, with a good amount of EDA out of the way,
# 1. Speed run an XGboost model with _only_the numeric variables 
#     on 5 fold cv.... submit as my baseline.
# 2. Bring in the categorical variables accordingly 
#     hoping to bring rmse down
# 3. Incorporate more data????
#   - Census Data (Populations)
#   - median househole income per state?


# Thoughts... 
# I'm starting to see why we should be pulling more data in.
# I'm seeing a whole subsection of states missing
# I've got a feeling that "Region" field won't be able to impart the necessary information
# that we'd need to do well on the testing set
# 


store_recipe <- recipe(profit ~ ., store_training) %>% 
  update_role(id, new_role = "id") %>% 
  step_rm(id, 
          postal_code,
          country,
          city,
          state) %>% 
  # step_other(city, threshold = 0.01) %>% 
  step_dummy(segment,
           # city,
           # state,
           region, 
           category, 
           sub_category) %>% 
  step_mutate(sales_per_quantity = sales / quantity) %>% 
  prep()


training <- bake(store_recipe, new_data =NULL )
testing  <- bake(store_recipe, new_data =store_test)

set.seed(42069) # such a hard decision
stores_folds <- vfold_cv(data = training, v = 10)

stores_xgb <- boost_tree(mode = "regression",
                         learn_rate = tune(),
                         trees = tune(),
                         mtry = tune(),
                         tree_depth = tune()
                         ) %>% 
  set_engine("xgboost")

stores_wkflow <- workflow() %>% 
  add_model(stores_xgb) %>% 
  add_formula(profit ~ .)

stores_metrics <- metric_set(rmse)

set.seed(42069)
xgb_param <- 
  stores_wkflow %>% 
  parameters() %>% 
  update(
    learn_rate = dials::learn_rate(),
    mtry = mtry(c(1L, 10L)),
    trees  = trees(),
    tree_depth = tree_depth()
  )

doParallel::registerDoParallel(10)
startingTime <- Sys.time()
stores_bayes <- tune_bayes(stores_wkflow,
   resamples = stores_folds,
   metrics = stores_metrics,
   param_info = xgb_param,
   initial = 10,
   iter = 30,
   control = control_bayes(verbose = TRUE))
Sys.time() - startingTime


stores_bayes %>% 
  autoplot(type = "performance") + 
  ylim(0, 1000)

stores_bayes %>% 
  autoplot(type = "parameters")


final_params <- stores_bayes %>% 
  select_best()

final_stores_xgb <- stores_xgb %>% 
  finalize_model(final_params)

final_fit <- final_stores_xgb %>% 
  fit(formula = profit ~ ., data = training)

final_preds <- final_fit %>% 
  predict(testing) %>% 
  bind_cols(store_test) %>% 
  select(id, profit = .pred)

final_preds %>% 
  write_csv(file.path(predictions_folder, "attempt9_bayes.csv"))
