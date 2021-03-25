library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(skimr)
library(memer)
library(janitor)
library(doParallel)


theme_set(theme_minimal())

data_folder <- 'data'
predictions_folder <- 'predictions'
models_folder <- 'models'
plots_folder <- 'plots'

views_data <- read_csv(file.path(data_folder, "s00e04-sliced_data.csv")) %>% 
  janitor::clean_names("small_camel") %>% 
  mutate(subtitle = ifelse(is.na(subtitle), "Missing", subtitle))
views_holdout <- read_csv(file.path(data_folder, "s00e04-holdout-data.csv")) %>% 
  janitor::clean_names("small_camel") %>% 
  mutate(subtitle = ifelse(is.na(subtitle), "Missing", subtitle))




skimr::skim(views_holdout)
# ── Data Summary ────────────────────────
# Values    
# Name                       views_data
# Number of rows             32661     
# Number of columns          17        
# _______________________              
# Column type frequency:               
#   character                3         
# numeric                  12        
# POSIXct                  2         
# ________________________             
# Group variables            None      
# 
# ── Variable type: character ─────────────────────────────────────────────────────────────────────────────────────────────────────
# skim_variable n_missing complete_rate   min   max empty n_unique whitespace
# 1 Title                 0         1         5    57     0    31123          0
# 2 Subtitle          12120         0.629     2   168     0    18856          0
# 3 Name                  0         1         3    31     0      216          0
# 
# ── Variable type: numeric ───────────────────────────────────────────────────────────────────────────────────────────────────────
# skim_variable          n_missing complete_rate          mean      sd    p0    p25    p50    p75    p100 hist 
# 1 Id                             0             1     570051.   3.91e 5     7 133059 624550 887157 1.22e 6 ▇▃▅▇▅
# 2 DaysSinceCreation              0             1        477.   3.97e 2     2    181    328    747 2.06e 3 ▇▂▂▁▁
# 3 DaysSinceLastUpdate            0             1        441.   3.80e 2     2    165    306    652 1.78e 3 ▇▃▂▁▁
# 4 TotalViews                     0             1       5092.   6.02e 4     0    138    413   1474 8.41e 6 ▇▁▁▁▁
# 5 TotalDownloads                 0             1        576.   5.09e 3     0      4     25    117 3.10e 5 ▇▁▁▁▁
# 6 TotalVotes                     0             1         15.1  1.21e 2     0      0      1      6 9.23e 3 ▇▁▁▁▁
# 7 TotalKernels                   0             1          5.93 6.31e 1     0      0      1      2 5.01e 3 ▇▁▁▁▁
# 8 DatasetTagCount                0             1       6159.   5.52e 3     4   1566   3636  10199 1.84e 4 ▇▂▃▁▂
# 9 CompetitionTagCount            0             1          7.95 1.77e 1     0      1      2      6 1.12e 2 ▇▁▁▁▁
# 10 KernelTagCount                 0             1       4284.   3.02e 4     0    322   1433   5280 1.27e 6 ▇▁▁▁▁
# 11 TotalCompressedBytes           0             1 3242723822.   4.26e11     0      0      0  31606 7.64e13 ▇▁▁▁▁
# 12 TotalUncompressedBytes         0             1 3221746902.   4.26e11     0      0      0  38894 7.64e13 ▇▁▁▁▁
# 
# ── Variable type: POSIXct ───────────────────────────────────────────────────────────────────────────────────────────────────────
# skim_variable n_missing complete_rate min                 max                 median              n_unique
# 1 CreationDate          0             1 2015-08-04 23:59:00 2021-03-20 11:49:29 2020-04-28 01:06:38    32650
# 2 VersionUpdate         0             1 2016-05-09 15:14:15 2021-03-20 11:49:29 2020-05-20 10:12:08    32638


# meme_get("HotlineDrake") %>%
#   meme_text_drake(top = "Four RStudio Panes",
#                   bot = "RIP Environment Pane")


# Correlation plot
# DataExplorer::plot_correlation(views_data)

# base R
hist(views_data$totalViews)

# looking quite right tailed... Poisson?
hist(log(views_data$totalViews))

# Hey-yo!

summary(views_data$totalViews)

# okay, with not a whole lot missing I'm going to:
# 1. speed run an XGboost model with a Poisson family
# 2. speed run an glm model with a Poisson family
# 3. confirm which model is better
# 4. SHAPs or coefficient value depending upon 3.

views_recipe <- recipe(totalViews ~ ., views_data) %>% 
  update_role(id, new_role = "ID") %>% 
  step_rm(creationDate, versionUpdate, title, subtitle, name, id) %>% 
  # step_range(all_predictors(), min = 0, max = 1) %>% 
  prep()


views_final <- bake(views_recipe, new_data = NULL)
final_holdout <- bake(views_recipe, new_data = views_holdout)


views_xgb_spec <- boost_tree(mode = "regression",
           mtry = tune(), 
           min_n = tune(),
           trees = 500,
           learn_rate = 0.3) %>% 
  set_engine("xgboost", objective = "count:poisson") 
  
views_glm_spec <- poisson_reg(mode = "regression",
                             penalty = tune(), 
                             mixture = tune()) %>% 
  set_engine("glmnet") 


views_glm_wf <- workflow() %>% 
  add_formula(totalViews ~ .) %>% 
  add_model(views_glm_spec)


set.seed(69420)
views_folds <- vfold_cv(data = views_final, v = 4)


doParallel::registerDoParallel(4)
views_grid <- tune_grid(views_wf,
  resamples = views_folds,
    control = control_grid(save_pred = T,
                           verbose = T))

views_grid_glm <- tune_grid(views_glm_wf,
  resamples = views_folds,
  control = control_grid(save_pred = T,
                         verbose = T))

xgb_results <- views_grid %>% 
  collect_metrics() %>% 
  mutate(model = "xgb")

glm_results <- views_grid_glm %>% 
  collect_metrics() %>% 
  mutate(model = "glm")

xgb_results %>% 
  bind_rows(glm_results) %>% 
  filter(.metric == "rsq") %>% 
  arrange(desc(mean))


# GBM with the best RSq for now... 
# destroying the GLM that's interesting
xgb_results %>% 
  filter(.metric == "rsq") %>% 
  ggplot(aes(x = mtry, y = min_n, color = mean)) +
    geom_point() +
    scale_color_continuous(type = "viridis") +
    labs(title = "Performance of R^2 on 4 fold cv",
         subtitle = "the lighter the circle, the better the R^2") +
    geom_smooth(method = "lm") +
    theme(plot.title.position = "plot")
    
# so, it seems the deeper the tree and min_n doesn't 
# appear to matter much


# let's finalize the workflow!

best_xgb_spec <- views_grid %>% 
  select_best(metric = "rsq")


best_xgb <- finalize_model(views_xgb_spec, best_xgb_spec) %>% 
  fit(totalViews ~ ., views_data)








