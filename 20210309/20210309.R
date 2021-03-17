
################## Nschampions2004 / Kyle  ######################
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

# alright....
## Gameplan is...
### 1. Setup pipeline so that we're not screwed with making preds
### 1. a. that was trash
### 2. Tune / model swap / bring in differing data set
### 3. Profit!
#### 4 .BAHAHAHAHAHAHAHAHAHAHAHAHAHAHAH



beer_data <- read.csv(file.path(data_folder, 'sliced_data.csv'),
                      na = c("", "NA", "  ")) %>%
  select(-review_aroma,-review_appearance, -review_palate, -review_taste,
         -brewery_city, -brewery_state) %>%
  mutate(across(dplyr::matches(missing_cols), tidyr::replace_na, "Missing"))

# Well at least this isn't missing any...
beer_holdout <- read.csv(file.path(data_folder, 'sliced_holdout_data.csv')) %>%
  select(-brewery_city, -brewery_state)

beer_holdout_clean <- read.csv(file.path(data_folder, 'sliced_holdout_data.csv'))

# simple checks to confirm we've got 100% of the data...
# a couple of our "review_overall" are missing like 2k
skimr::skim(beer_data)
# ── Variable type: character ──────────────────────────────────────────────────────────────────────────────────────────────────────
# skim_variable      n_missing complete_rate   min   max empty n_unique whitespace
# 1 brewery_city               0             1     0    46   480     2024          0
# 2 brewery_state              0             1     0     3 12676       65          0
# 3 brewery_country            0             1     0     2     5      112          0
# 4 brewery_name               0             1     3    66     0     2856          0
# 5 review_profilename         0             1     3    16     0     8905          0
# 6 beer_style                 0             1     4    35     0      104          0
# 7 beer_name                  0             1     2    74     0    12079          0
# 8 beer_category              0             1     3    26     0       47          0
# 9 beer_availability          0             1     4    22     0       15          0
#
# ── Variable type: numeric ────────────────────────────────────────────────────────────────────────────────────────────────────────
# skim_variable                        n_missing complete_rate          mean           sd            p0           p25
# 1 brewery_id                                   0         1           3047.       5561.             1           144
# 2 review_time                                  0         1     1221394435.   77516362.     885427201    1169015245.
# 3 review_overall                               0         1              3.76        0.743          1             3.5
# 4 beer_abv                                  2266   .05          5.1
# 5 beer_beerid                                  0         1          22675.      22188.             4          1881
# 6 brewery_review_time_mean                     0         1     1221056274.   35582440.     940818342.   1199709009.
# 7 brewery_review_overall_mean                  0         1              3.76        0.334          1.44          3.65
# 8 brewery_review_aroma_mean                    0         1              3.67        0.409          1.25          3.56
# 9 brewery_review_appearance_mean               0         1              3.79        0.328          1.5           3.71
# 10 brewery_review_palate_mean                   0         1              3.68        0.361          1.12          3.57
# 11 brewery_review_taste_mean                    0         1              3.72        0.403          1.38          3.60
# 12 brewery_beer_abv_mean                      194         0.996          6.75        1.38           0.5           5.74
# 13 beer_category_review_time_mean               0         1     1221170284.   13703567.    1182969396.   1212239293.
# 14 beer_category_review_overall_mean            0         1              3.77        0.236          2.69          3.75
# 15 beer_category_review_aroma_mean              0         1              3.67             0.955          6.76        2.14           0 0.345          2.15          3.56
# 16 beer_category_review_appearance_mean         0         1              3.79        0.288          2.41          3.71
# 17 beer_category_review_palate_mean             0         1              3.68        0.285          2.33          3.60
# 18 beer_category_review_taste_mean              0         1              3.73        0.319          2.26          3.62
# 19 beer_category_beer_abv_mean                  0         1              6.74        1.48           2.08          5.44
# p50           p75          p100 hist
# 1        454          2267         27945    ▇▁▁▁▁
# 2 1236212026.   1286883567.   1326266737    ▁▁▂▅▇
# 3          4             4             5    ▁▁▂▇▃
# 4          6.1           8            41    ▇▂▁▁▁
# 5      15452.        40923.        77291    ▇▂▂▂▁
# 6 1219724963.   1242786160.   1325804120    ▁▁▁▇▂
# 7          3.82          3.98          5    ▁▁▂▇▁
# 8          3.77          3.93          5    ▁▁▂▇▁
# 9          3.87          4             5    ▁▁▁▇▁
# 10          3.77          3.92          5    ▁▁▂▇▁
# 11          3.83          3.99          5    ▁▁▂▇▁
# 12          6.54          7.73         13    ▁▃▇▂▁
# 13 1219863494.   1230993593.   1277199747.   ▁▇▅▃▁
# 14          3.81          3.91          4.11 ▁▁▁▆▇
# 15          3.75          3.91          4.15 ▁▁▁▆▇
# 16          3.89          3.97          4.09 ▁▁▁▅▇
# 17          3.78          3.88          4.07 ▁▁▁▆▇
# 18          3.83          3.95          4.16 ▁▁▁▅▇
# 19          6.58          7.68         10.8  ▁▇▅▆▂

# knitting... the most I'll do tonight
DataExplorer::create_report(beer_data)

DataExplorer::plot_correlation(beer_data)

# address the missing in the target
beer_recipe <-recipe(review_overall ~ ., beer_data) %>%
  step_meanimpute(review_overall) %>%
  step_meanimpute(beer_abv) %>%
  step_knnimpute(brewery_beer_abv_mean) %>%
  update_role(beer_beerid, new_role = "ID") %>%
  step_other(c(beer_style, beer_category, beer_availability), threshold = 0.01) %>%
  step_other(c(brewery_country,
             brewery_name), threshold = 0.01) %>%  #this might have to be really low
  step_rm(all_nominal()) %>%
  # step_dummy(brewery_city, one_hot = T) %>%
  # step_dummy(brewery_state, one_hot = T) %>%
  # step_dummy(brewery_country, one_hot = T) %>%
  # step_dummy(brewery_name, one_hot = T) %>%
  # step_dummy(beer_style, one_hot = T) %>%
  # step_dummy(beer_category, one_hot = T) %>%
  # step_dummy(beer_availability, one_hot = T) %>%
  prep()


beer_training <- bake(beer_recipe, new_data = NULL) %>%
  na.omit() # GRRR
beer_testing <- bake(beer_recipe, new_data = beer_holdout)


# much more manageable in terms of number of features.
skim(beer_training)
# ── Data Summary ────────────────────────
# Values
# Name                       Piped data
# Number of rows             3435
# Number of columns          25
# _______________________
# Column type frequency:
#   factor                   7
# numeric                  18
# ________________________
# Group variables            None
#
# ── Variable type: factor ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
# skim_variable     n_missing complete_rate ordered n_unique top_counts
# 1 brewery_city              0             1 FALSE          1 "oth: 3435, Bou: 0, Bro: 0, Cha: 0"
# 2 brewery_state             0             1 FALSE          1 "oth: 3435, CA: 0, CO: 0, GA: 0"
# 3 brewery_country           0             1 FALSE          1 "US: 3435, BE: 0, CA: 0, DE: 0"
# 4 brewery_name              0             1 FALSE          1 "oth: 3435, Anh: 0, Bos: 0, Gre: 0"
# 5 beer_style                0             1 FALSE          7 "oth: 1067, Ame: 718, Rus: 602, Sai: 521"
# 6 beer_category             0             1 FALSE         10 "bro: 751, sto: 724, ipa: 522, Sai: 521"
# 7 beer_availability         0             1 FALSE          8 " Ye: 1155,  Fa: 882,  Ro: 560, oth: 355"
#
# ── Variable type: numeric ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# skim_variable                        n_missing complete_rate          mean           sd            p0           p25           p50
# 1 brewery_id                                   0             1      13014           0          13014         13014         13014
# 2 review_time                                  0             1 1259739425.   43227882.    1139025983    1226750210    1268434426
# 3 beer_abv                                     0             1          7.22        2.20           4             5.1           6.7
# 4 beer_beerid                                  0             1      36475.       8231.         28165         30845         33127
# 5 brewery_review_time_mean                     0             1 1260451483.          0     1260451483.   1260451483.   1260451483.
# 6 brewery_review_overall_mean                  0             1          4.25        0              4.25          4.25          4.25
# 7 brewery_review_aroma_mean                    0             1          4.15        0              4.15          4.15          4.15
# 8 brewery_review_appearance_mean               0             1          4.16        0              4.16          4.16          4.16
# 9 brewery_review_palate_mean                   0             1          4.16        0              4.16          4.16          4.16
# 10 brewery_review_taste_mean                    0             1          4.23        0              4.23          4.23          4.23
# 11 brewery_beer_abv_mean                        0             1          7.24        0              7.24          7.24          7.24
# 12 beer_category_review_time_mean               0             1 1227198347.   10882397.    1211812312.   1217347809.   1230993593.
# 13 beer_category_review_overall_mean            0             1          3.83        0.110          3.56          3.77          3.91
# 14 beer_category_review_aroma_mean              0             1          3.76        0.162          3.48          3.61          3.90
# 15 beer_category_review_appearance_mean         0             1          3.90        0.144          3.64          3.78          3.97
# 16 beer_category_review_palate_mean             0             1          3.75        0.135          3.54          3.61          3.87
# 17 beer_category_review_taste_mean              0             1          3.81        0.136          3.59          3.70          3.90
# 18 beer_category_beer_abv_mean                  0             1          6.67        0.985          5.37          5.59          6.91
# p75          p100 hist
# 1      13014         13014    ▁▁▇▁▁
# 2 1296938465    1326164886    ▁▃▆▆▇
# 3          9            10.3  ▇▂▃▅▆
# 4      39917         69307    ▇▂▂▁▁
# 5 1260451483.   1260451483.   ▁▁▇▁▁
# 6          4.25          4.25 ▁▁▇▁▁
# 7          4.15          4.15 ▁▁▇▁▁
# 8          4.16          4.16 ▁▁▇▁▁
# 9          4.16          4.16 ▁▁▇▁▁
# 10          4.23          4.23 ▁▁▇▁▁
# 11          7.24          7.24 ▁▁▇▁▁
# 12 1239876689.   1251253839.   ▇▁▃▆▁
# 13          3.92          3.93 ▁▁▅▁▇
# 14          3.90          3.91 ▃▃▁▁▇
# 15          3.97          4.09 ▂▇▁▇▅
# 16          3.87          3.88 ▂▅▁▁▇
# 17          3.92          3.95 ▃▃▁▁▇
# 18          7.50          8.02 ▇▁▅▁▇



# modelling, just wanna to get something going... getting sick of looking at skimr... you probably are too
beer_lm <- linear_reg(
  mode = "regression",
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet")

beer_gbm <- boost_tree(
  mode = "regression",
  trees = 1000,
  mtry = 1,
  min_n = 1,
  tree_depth = 4,
  stop_iter = 25,
  loss_reduction = 0,
  learn_rate = 0.3,
) %>%
  set_engine("xgboost")


beer_lm_wf <- workflow() %>%
  add_model(beer_lm) %>%
  add_formula(review_overall ~ . -brewery_id)

beer_gbm_wf <- workflow() %>%
  add_model(beer_gbm) %>%
  add_formula(review_overall ~ . -brewery_id)


set.seed(69420)
beer_folds <- vfold_cv(data = beer_training)

registerDoParallel(8)
beer_lm_grid <- tune_grid(object = beer_lm_wf,
    resamples = beer_folds,
    grid = 4,
    control = control_grid(save_pred = T, verbose = T)
)

beer_lm_grid <- tune_grid(object = beer_gbm_wf,
    resamples = beer_folds,
    grid = 4,
    control = control_grid(save_pred = T, verbose = T)
)

# so, I'm really hoping this finishes in time :(
# I don't really like to rely on Hail Mary's...
# this has been a fun process
# if I were to do this over again, I would have not tried to
# eat the whole pie with all the variables.
# Screw it

beer_gbm_grid <- beer_lm_grid

beer_lm_spec <- beer_lm_grid %>%
  select_best("rmse")

beer_gbm_spec <- beer_gbm_grid %>%
  select_best("rmse")

gbm <- beer_gbm %>%
  fit(review_overall ~ . , beer_training)

gbm %>%
  predict(beer_training) %>%
  rename(review_overall_pred = .pred)
  bind_cols(beer_training) %>%
  ggplot(aes(x = review_overall_pred, y = review_overall)) +
    geom_jitter()

gbm %>%
  predict(beer_holdout) %>%
  bind_cols(beer_holdout) %>%
  rename(review_overall = .pred) %>%
  left_join(beer_holdout_clean) %>%
  write_csv(file.path(predictions_folder, 'gbm_preds2.csv'))


beer_holdout_clean %>%
  nrow()
# GOOD!!!
preds %>%
  nrow()

final_lm <- finalize_model(beer_lm, beer_lm_spec) %>%
  fit(formula = review_overall ~ ., data = beer_training)

# predictions versus actuals
final_lm %>%
  predict(beer_training) %>%
  bind_cols(beer_training) %>%
  ggplot(aes(x = .pred, y = review_overall)) +
    geom_jitter()


lm_preds <- final_lm %>%
  predict(beer_holdout) %>%
  bind_cols(beer_holdout) %>%
  rename(review_overall = .pred) %>%
  left_join(beer_holdout_clean) %>%
  write_csv(file.path(predictions_folder, 'lm_preds.csv'))

