library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(doParallel)
library(skimr)

data_folder <- "data"
plots_folder <- "plots"
predictions_folder <- 'predictions'
models_folder <- 'models'

theme_set(theme_minimal())

missing_cols <- c("brewery_city", "brewery_state", "brewery_country")

brewery_and_beer_info <- read_csv(file.path(data_folder, 'brewery_and_beer_info.csv'))
reviewer_info <- read_csv(file.path(data_folder, 'reviewer_info.csv'))

beer_data <- read_csv(file.path(data_folder, 'sliced_data.csv'),
                      na = c("", "NA", "  ")) %>%
  select(-review_aroma,-review_appearance, -review_palate, -review_taste) %>%
  mutate(across(dplyr::matches(missing_cols), tidyr::replace_na, "Missing")) %>%
  left_join(reviewer_info) %>%
  left_join(brewery_and_beer_info)

beer_holdout <- read_csv(file.path(data_folder, 'sliced_holdout_data.csv')) %>%
  left_join(reviewer_info) %>%
  left_join(brewery_and_beer_info)


cols <- null_cols(beer_data)

null_cols <- function(x) {
  "Return the columns with nulls in them and their corresponding type"

  cols <- names(x)[unlist(map(x, ~sum(is.na(.))>0))]

  pull <- x %>% select(any_of(cols))

  unlist(map(pull, typeof))
}

other_threshold <- function(x) {
  # if (typeof(x) %in% c("character", "factor"))
  #   glue::glue({})

  # browser()
  # I need:
  # 1. to find the names of character columns
  # 2. Their cardinality DONE
  # 3. Other Suggest Threshold

  unique_vals <- length(unique(x))

  # for (i in seq(from = 0.0, to = 0.10, by = 0.01)) {
  #
  # }
}


beer_recipe <-recipe(review_overall ~ ., beer_data) %>%
  step_meanimpute(all_numeric()) %>%
  update_role(beer_beerid, new_role = "ID") %>%
  step_other(beer_style, threshold = 0.02) %>% #good
  step_other(beer_name, threshold = 0.002) %>%  # good
  step_other(c(beer_category, beer_availability), threshold = 0.01) %>%
  step_other(brewery_state, threshold = 0.1) %>%
  step_other(review_profilename, threshold = 0.007) %>%
  step_other(brewery_city, threshold = 0.01) %>%
  step_other(c(brewery_country,
               brewery_name), threshold = 0.01) %>%  #this might have to be really low
  step_dummy(brewery_city) %>%
  step_dummy(brewery_state) %>%
  step_dummy(brewery_country) %>%
  step_dummy(brewery_name) %>%
  step_dummy(beer_style) %>%
  step_dummy(beer_category) %>%
  step_dummy(beer_availability) %>%
  step_dummy(beer_name) %>%
  step_dummy(review_profilename) %>%
  prep()


final_training <- bake(beer_recipe, NULL) %>%
  janitor::clean_names()

null_cols(final_training)

final_holdout <- bake(beer_recipe, beer_holdout) %>%
  janitor::clean_names()

beer_lm <- linear_reg(mode = "regression",
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet")

beer_gbm <- boost_tree(
    mode = "regression",
    min_n = tune(),
    mtry = tune(),
    trees = 1000,
    learn_rate = 0.3,
    loss_reduction = 0,
    sample_size = 1,
    stop_iter = 50
) %>%
  set_engine("xgboost")


lm_workflow <- workflow() %>%
  add_model(beer_lm) %>%
  add_formula(review_overall ~ .)

gbm_workflow <- workflow() %>%
  add_model(beer_gbm) %>%
  add_formula(review_overall ~ .)

set.seed(69420)
beer_resamples <- vfold_cv(final_training, v = 10)

doParallel::registerDoParallel(8)
lm_grid <- tune_grid(
  lm_workflow,
  resamples = beer_resamples,
  grid = 10,
  control = control_grid(verbose = T, save_pred = T)
)

gbm_grid <- tune_grid(
  gbm_workflow,
  resamples = beer_resamples,
  grid = 10,
  control = control_grid(verbose = T, save_pred = T)
)


gbm_grid %>%
  collect_metrics() %>%
  inner_join(gbm_grid %>% select_best("rmse"))

lm_grid %>%
  collect_predictions() %>%
  filter(id == "Fold01",
         .config == "Preprocessor1_Model01") %>%
  ggplot(aes(x = .pred, y = review_overall)) +
    geom_point() +
    geom_smooth(method = "lm")

gbm_grid %>%
  collect_predictions() %>%
  filter(id == "Fold01",
         .config == "Preprocessor1_Model01") %>%
  ggplot(aes(x = .pred, y = review_overall)) +
  geom_point() +
  geom_smooth(method = "lm")



lm_grid %>%
  collect_metrics() %>%
  ggplot(aes(x = penalty, y = mixture, fill = mean, color = mean)) +
    geom_point() +
    geom_text(aes(label = round(mean, 4)), nudge_x = 0.07) +
    facet_wrap(~.metric)


# high mixture towards L2 with a weak penalty seems best.

lm_spec <- lm_grid %>%
  select_best("rmse")

final_lm <- finalize_model(x = beer_lm, lm_spec) %>%
  fit(formula = review_overall ~ ., data = final_training)

lm_grid %>%
  collect_metrics() %>%
  inner_join(lm_spec, by = c("penalty", "mixture", ".config"))





