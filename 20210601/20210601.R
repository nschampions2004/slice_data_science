library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(skimr)
library(memer)
library(janitor)
library(doParallel)
library(poissonreg)

source("utils.R")

theme_set(theme_minimal())

data_folder <- 'data'
predictions_folder <- 'predictions'
models_folder <- 'models'
plots_folder <- 'plots'

train_raw <- read_csv(file.path(data_folder, 'train.csv'),
                      col_types = cols(
                        .default = col_character(),
                        game_id = col_double(),
                        min_players = col_double(),
                        max_players = col_double(),
                        avg_time = col_double(),
                        min_time = col_double(),
                        max_time = col_double(),
                        year = col_double(),
                        geek_rating = col_double(),
                        num_votes = col_double(),
                        age = col_double(),
                        owned = col_double(),
                        category9 = col_character(),
                        category10 = col_character(),
                        category11 = col_character(),
                        category12 = col_character()
                      )) %>% 
  janitor::clean_names("small_camel")

test_raw <- read_csv(file.path(data_folder, 'test.csv')) %>% 
  janitor::clean_names("small_camel")  
                      

skimr::skim(train_raw)
# ── Data Summary ────────────────────────
# Values   
# Name                       train_raw
# Number of rows             3499     
# Number of columns          26       
# _______________________             
# Column type frequency:              
#   character                15       
# numeric                  11       
# ________________________            
# Group variables            None     
# 
# ── Variable type: character ──────────────────────────────────────────────────────────────────────────────────────
# skim_variable n_missing complete_rate   min   max empty n_unique whitespace
# 1 names                 0      1            1    82     0     3485          0
# 2 mechanic              0      1            4   258     0     1758          0
# 3 category1             0      1            4    26     0       78          0
# 4 category2           611      0.825        4    26     0       81          0
# 5 category3          1773      0.493        4    26     0       72          0
# 6 category4          2636      0.247        4    25     0       62          0
# 7 category5          3098      0.115        4    25     0       45          0
# 8 category6          3363      0.0389       4    25     0       38          0
# 9 category7          3453      0.0131       4    25     0       21          0
# 10 category8          3480      0.00543      5    19     0       14          0
# 11 category9          3494      0.00143      7    19     0        5          0
# 12 category10         3495      0.00114      6    25     0        3          0
# 13 category11         3498      0.000286     9     9     0        1          0
# 14 category12         3498      0.000286    15    15     0        1          0
# 15 designer              0      1            4   157     0     1905          0
# 
# ── Variable type: numeric ────────────────────────────────────────────────────────────────────────────────────────
# skim_variable n_missing complete_rate     mean        sd       p0      p25      p50       p75      p100 hist 
# 1 gameId                0             1 89632.   77041.        2    11164.   73538    160677    244522    ▇▂▃▃▂
# 2 minPlayers            0             1     2.01     0.666     0        2        2         2         8    ▂▇▁▁▁
# 3 maxPlayers            0             1     5.06     7.24      0        4        4         6       200    ▇▁▁▁▁
# 4 avgTime               0             1   117.     488.        0       30       60       120     22500    ▇▁▁▁▁
# 5 minTime               0             1    82.5    214.        0       30       45        90      5400    ▇▁▁▁▁
# 6 maxTime               0             1   117.     488.        0       30       60       120     22500    ▇▁▁▁▁
# 7 year                  0             1  1996.     162.    -3000     2003     2011      2015      2018    ▁▁▁▁▇
# 8 geekRating            0             1     6.09     0.483     5.64     5.73     5.90      6.30      8.50 ▇▂▁▁▁
# 9 numVotes              0             1  2006.    4645.       62      281      618      1640     77423    ▇▁▁▁▁
# 10 age                   0             1    10.4      3.22      0        8       11        12        42    ▃▇▁▁▁
# 11 owned                 0             1  3055.    6369.       49      622.    1204      2723    111807    ▇▁▁▁▁


train_raw %>% 
  count(year)

# so, did you see what I did here?  
# this section unpivots the category field
# it'll get us to binary flags of the 
# category field
category_count_training <- train_raw %>% 
  select(gameId, category1:category12) %>% 
  pivot_longer(cols = category1:category12,
               names_to = "to_drop", 
               values_to = "category") %>% 
  filter(!is.na(category)) %>% 
  select(-to_drop) %>% 
  mutate(category = stringr::str_to_lower(category),
         count = 1) 

# push testing through the same process 
category_count_testing <- test_raw %>% 
  select(gameId, category1:category12) %>% 
  pivot_longer(cols = category1:category12,
               names_to = "to_drop", 
               values_to = "category") %>% 
  filter(!is.na(category)) %>% 
  select(-to_drop) %>% 
  mutate(category = stringr::str_to_lower(category),
         count = 1) 


# quick check to make sure there aren't categories missing
# in training or testing
category_count_training %>% 
  distinct(category) %>% 
  anti_join(category_count_testing %>% distinct(category))
# expect 0 rows returned
# ... and zero rows returned.
# sooooooo satisfying !!!!! CHICKENS IN CHAT!!!!!!

# now this pivots the categories back up to columns 
# making them binary flag columns
train_category <- category_count_training %>%
  arrange(category) %>% 
  pivot_wider(id_cols = gameId,
              names_from = category,
              values_from = count,
              values_fill = 0) %>% 
  janitor::clean_names("small_camel")

test_category <- category_count_testing %>% 
  arrange(category) %>% 
  pivot_wider(id_cols = gameId,
              names_from = category,
              values_from = count,
              values_fill = 0) %>% 
  janitor::clean_names("small_camel")

train_mechanic <- train_raw %>% 
  select(gameId, mechanic)

unique_trains <- train_mechanic %>% 
  separate(col = mechanic, 
           into = c("mech1", "mech2", "mech3", 
                    "mech4", "mech5", "mech6", 
                    "mech7", "mech8", "mech9",
                    "mech10", "mech11", "mech12"),
           sep = ", ", 
           extra = "warn", 
           fill = "right") %>% 
  pivot_longer(cols = mech1:mech12, 
               names_to = "to_drop",
               values_to = "mechanic" ) %>% 
  dplyr::filter(!is.na(mechanic)) %>% 
  distinct(mechanic)


unique_tests <- test_raw %>% 
  # filter(gameId != 175878) %>% 
  separate(col = mechanic, 
           into = c("mech1", "mech2", "mech3", 
                    "mech4", "mech5", "mech6", 
                    "mech7", "mech8", "mech9",
                    "mech10", "mech11", "mech12",
                    "mech13", "mech14", "mech15",
                    "mech16", "mech17", "mech18",
                    "mech19"),
           sep = ", ", 
           extra = "warn", 
           fill = "right") %>% 
  pivot_longer(cols = mech1:mech12, 
               names_to = "to_drop",
               values_to = "mechanic" ) %>% 
  dplyr::filter(!is.na(mechanic)) %>% 
  distinct(mechanic)
  

unique_trains %>% 
  anti_join(unique_tests)
# we'll have to add "Singing" to testing data

testing_mechanic <- test_raw %>% 
  select(gameId, mechanic) %>% 
  separate(col = mechanic, 
           into = c("mech1", "mech2", "mech3", 
                    "mech4", "mech5", "mech6", 
                    "mech7", "mech8", "mech9",
                    "mech10", "mech11", "mech12",
                    "mech13", "mech14", "mech15",
                    "mech16", "mech17", "mech18",
                    "mech19"),
           sep = ", ", 
           extra = "warn", 
           fill = "right") %>% 
  pivot_longer(cols = mech1:mech19, 
               names_to = "to_drop",
               values_to = "mechanic")  %>% 
  dplyr::filter(!is.na(mechanic)) %>% 
  select(-to_drop) %>% 
  mutate(mech_flag = 1) %>% 
  arrange(mechanic) %>% 
  pivot_wider(id_cols = gameId, 
              names_from = mechanic,
              values_from = mech_flag,
              values_fill = 0) %>% 
  janitor::clean_names("small_camel") %>% 
  mutate(singing = 0)

training_mechanic <- train_raw %>% 
  select(gameId, mechanic) %>% 
  separate(col = mechanic, 
           into = c("mech1", "mech2", "mech3", 
                    "mech4", "mech5", "mech6", 
                    "mech7", "mech8", "mech9",
                    "mech10", "mech11", "mech12",
                    "mech13", "mech14", "mech15",
                    "mech16", "mech17", "mech18",
                    "mech19"),
           sep = ", ", 
           extra = "warn", 
           fill = "right") %>% 
  pivot_longer(cols = mech1:mech19, 
               names_to = "to_drop",
               values_to = "mechanic")  %>% 
  dplyr::filter(!is.na(mechanic)) %>% 
  select(-to_drop) %>% 
  mutate(mech_flag = 1) %>%
  arrange(mechanic) %>% 
  pivot_wider(id_cols = gameId, 
              names_from = mechanic,
              values_from = mech_flag,
              values_fill = 0) %>% 
  janitor::clean_names("small_camel")



# basic recipe to clean the data up
geek_rec <- recipes::recipe(geekRating ~ . , data = train_raw) %>% 
  update_role(gameId, new_role = "ID") %>%
  step_rm(category1:category12) %>% 
  prep()

# bring back in the category fields on gameId
training <- bake(geek_rec, new_data = NULL) %>% 
  left_join(train_category, by = "gameId") %>% 
  select(-mechanic, 
         -designer,
         -names) %>% 
  left_join(training_mechanic, by = "gameId")

testing <- bake(geek_rec, new_data = test_raw) %>% 
  left_join(test_category, by = "gameId") %>% 
  select(-mechanic, -names) %>% 
  left_join(testing_mechanic, by = "gameId")


setdiff(names(training), names(testing))

# quick correlation plot nothing really lining up with geek rating
DataExplorer::plot_correlation(training)

# setup an xgboost model
geek_xgb_spec <- boost_tree(mode = "regression",
                            tree_depth = tune(),
                            learn_rate = tune(),
                            trees = tune(),
                            mtry = tune(),
                            min_n = tune()) %>% 
  set_engine("xgboost") 

# set up some vfolds
set.seed(42069)
geek_vfolds <- vfold_cv(data = training, v = 2)

# set up workflow
geek_xgb_wf <- workflow() %>% 
  add_formula(geekRating ~ .) %>% 
  add_model(geek_xgb_spec)

# a grid search
doParallel::registerDoParallel(10)
geek_grid <- tune_grid(geek_xgb_wf,
 resamples = geek_vfolds,
 # control = control_grid(save_pred = TRUE,
 #                        verbose = TRUE),
 metrics = metric_set(rmse),
 grid = crossing(tree_depth = c(3), 
                 learn_rate = c(0.2),
                 trees = c(1200, 2400, 4800),
                 mtry = c(0.8),
                 min_n = c(1)))


autoplot(geek_grid)

# 6, 0.2 -> 0.1715 rmse, tree_depth = c(3, 6), learn_rate = c(0.1, 0.2, 0.3),
# 6, 0.2-> 0.1715 rmse tree_depth = c(6, 12), learn_rate = c(0.1, 0.2),
# adding mechanic, , -> rmse, tree_depth = c(3, 6), learn_rate = c(0.1, 0.2, 0.3),


best_spec <- select_best(geek_grid)

final_fit <- geek_xgb_wf %>% 
  finalize_workflow(best_spec) %>% 
  fit(training)

final_preds <- final_fit %>% 
  predict(testing) %>% 
  bind_cols(testing) %>% 
  select(game_id = gameId, geek_rating = .pred) %>% 
  write_csv(file.path('predictions', 'submission1.csv'))


