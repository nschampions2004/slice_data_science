library(tidyverse)
library(tidymodels)
library(skimr)
library(DataExplorer)

data_folder <- 'data'
plots_folder <- 'plots'
models_folder <- 'models'
predictions_folder <- "predictions"

train_raw <- read_csv(file.path(data_folder, 'train.csv')) %>% 
  mutate_if(is.character, as.factor) 
test_raw <- read_csv(file.path(data_folder, 'test.csv')) %>% 
  mutate_if(is.character, as.factor)

skimr::skim(train_raw)
# ── Data Summary ────────────────────────
# Values   
# Name                       train_raw
# Number of rows             21000    
# Number of columns          34       
# _______________________             
# Column type frequency:              
#   character                17       
# numeric                  17       
# ________________________            
# Group variables            None     
# 
# ── Variable type: character ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# skim_variable    n_missing complete_rate   min   max empty n_unique whitespace
# 1 operator_id              0         1         3     5     0      276          0
# 2 operator                 0         1         3    33     0      275          0
# 3 aircraft                 0         1         3    20     0      424          0
# 4 aircraft_type         4992         0.762     1     1     0        2          0
# 5 aircraft_make         5231         0.751     2     3     0       62          0
# 6 engine_type           5703         0.728     1     3     0        8          0
# 7 airport_id               0         1         3     5     0     1039          0
# 8 airport                 34         0.998     4    53     0     1038          0
# 9 state                 2664         0.873     2     2     0       60          0
# 10 faa_region            2266         0.892     3     3     0       14          0
# 11 flight_phase          6728         0.680     4    12     0       12          0
# 12 visibility            7699         0.633     3     7     0        5          0
# 13 precipitation        10327         0.508     3    15     0        8          0
# 14 species_id               0         1         1     6     0      447          0
# 15 species_name             7         1.00      4    50     0      445          0
# 16 species_quantity       532         0.975     1     8     0        4          0
# 17 flight_impact         8944         0.574     4    21     0        6          0
# 
# ── Variable type: numeric ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
# skim_variable    n_missing complete_rate       mean       sd    p0   p25    p50    p75  p100 hist 
# 1 id                       0        1      14981.     8663.        1 7459. 14978. 22472. 30000 ▇▇▇▇▇
# 2 incident_year            0        1       2006.        6.72   1990 2001   2007   2012   2015 ▂▃▅▆▇
# 3 incident_month           0        1          7.19      2.79      1    5      8      9     12 ▃▅▆▇▆
# 4 incident_day             0        1         15.6       8.82      1    8     15     23     31 ▇▇▇▇▆
# 5 aircraft_model        6259        0.702     24.6      21.7       0   10     22     37     98 ▇▆▂▁▁
# 6 aircraft_mass         5694        0.729      3.50      0.887     1    3      4      4      5 ▁▁▂▇▁
# 7 engine_make           6155        0.707     21.2      11.0       1   10     22     34     47 ▇▂▆▇▁
# 8 engine_model          6337        0.698     10.0      12.9       1    1      4     10     91 ▇▁▁▁▁
# 9 engines               5696        0.729      2.05      0.464     1    2      2      2      4 ▁▇▁▁▁
# 10 engine1_position      5838        0.722      2.99      2.09      1    1      1      5      7 ▇▁▂▅▁
# 11 engine2_position      6776        0.677      2.91      2.01      1    1      1      5      7 ▇▁▂▅▁
# 12 engine3_position     19676        0.0630     3.02      1.95      1    1      4      5      5 ▇▁▁▁▇
# 13 engine4_position     20650        0.0167     2.02      1.43      1    1      1      4      5 ▇▁▁▃▁
# 14 height                8469        0.597    819.     1773.        0    0     50    800  24000 ▇▁▁▁▁
# 15 speed                12358        0.412    141.       52.3       0  120    137    160   2500 ▇▁▁▁▁
# 16 distance              8913        0.576      0.663     3.33      0    0      0      0    100 ▇▁▁▁▁
# 17 damaged                  0        1          0.0857    0.280     0    0      0      0      1 ▇▁▁▁▁



plane_rec <-  recipe(damaged ~ ., train_raw) %>% 
  update_role(id, new_role = "id") %>% 
  step_meanimpute(all_numeric()) %>%
  step_knnimpute(operator_id, 
                 operator,
                 aircraft,
                 aircraft_make,
                 airport_id,
                 airport,
                 state,
                 species_id,
                 species_name,
                 faa_region,
                 flight_phase,
                 visibility,
                 precipitation,
                 species_quantity,
                 flight_impact) %>% 
  # step_unknown(all_nominal(), new_level = "unknown") %>%
  # step_other(operator_id, threshold = 0.1) %>% 
  # step_other(operator, threshold = 0.1) %>% 
  # step_other(aircraft, threshold = 0.05) %>%
  # step_other(aircraft_make, threshold = 0.05) %>% 
  # step_other(airport_id, threshold = 0.05) %>%
  # step_other(airport, threshold = 0.02) %>% 
  # step_other(state, threshold = 0.05) %>%
  # step_other(species_id, threshold = 0.08) %>% 
  # step_other(species_name, threshold = 0.08) %>% 
  step_rm(id) %>% 
  prep()

training <- bake(plane_rec, new_data = NULL) %>%
  mutate(damaged = as.factor(damaged))
testing <- bake(plane_rec, new_data = test_raw)


# unique(training$operator_id)
# unique(training$operator)
# unique(training$aircraft)
# unique(training$aircraft_make)
# unique(training$airport_id)
# unique(training$airport)
# unique(training$state)
# unique(training$species_id)
# unique(training$species_name)


DataExplorer::plot_correlation(training)

plane_lr <- logistic_reg(penalty = tune(), 
                        mixture = tune()) %>% 
  set_engine("glmnet")
  
set.seed(42069)
plan_folds <- vfold_cv(data = training, 
                       v = 5, 
                       strata = damaged)  

plan_wkflow <- workflow() %>% 
  add_model(plane_lr) %>% 
  add_formula(damaged ~ .)

plane_metric_set <- metric_set(mn_log_loss)

doParallel::registerDoParallel(5)
plane_grid <- tune_grid(
  plan_wkflow,
  grid = 10,
  resamples = plan_folds,
  metrics = plane_metric_set,
  control = control_grid(verbose = TRUE, save_pred = TRUE)
)

autoplot(plane_grid)

plane_grid %>% 
  collect_metrics() %>% 
  arrange(mean)

lowest_mn_log_loss <- select_best(plane_grid, 
                                  metric = "mn_log_loss")

final_plane_lr <- plane_lr %>% 
  finalize_model(lowest_mn_log_loss) %>% 
  fit(data = training,
      formula = damaged ~ .) 

pred_final_lm <- final_plane_lr %>% 
  predict(testing) %>% 
  mutate(.pred_class = ifelse(is.na(.pred_class), 0, 1))


pred_final_lm %>%
  rename(damaged = .pred_class) %>% 
  bind_cols(id=test_raw$id) %>% 
  write_csv(file.path(predictions_folder, "attempt1_lr_0.003_0.0735.csv"))







# xgboost -----------------------------------------------------------------
plane_xgb <- boost_tree(learn_rate = tune(), 
                        trees = tune(),
                        tree_depth = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

set.seed(42069)
plan_xgb_wkflow <- workflow() %>% 
  add_model(plane_xgb) %>% 
  add_formula(damaged ~ .)

plane_metric_set <- metric_set(mn_log_loss)

doParallel::registerDoParallel(5)
plane_grid <- tune_grid(
  plan_xgb_wkflow,
  grid = crossing(trees = c(100, 300, 500),
                  learn_rate = c(0.10, 0.2),
                  tree_depth = c(3, 5, 7)),
  resamples = plan_folds,
  metrics = plane_metric_set,
  control = control_grid(verbose = TRUE, save_pred = TRUE)
)

autoplot(plane_grid)

final_xgb <- plane_xgb %>% 
  finalize_model(select_best(plane_grid)) %>% 
  fit(data = training,
      formula = damaged ~ .) 

pred_final_xgb <- final_xgb %>% 
  predict(testing)

pred_final_xgb %>%
  rename(damaged = .pred_class) %>% 
  bind_cols(id=test_raw$id) %>% 
  write_csv(file.path(predictions_folder, "attempt4_xgb_0.1_300_3.csv"))


