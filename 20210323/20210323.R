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
# 1. speed run an XGboost model with a Poisson family.  DONE!
# 2. speed run an glm model with a Poisson family DONE!
# 3. confirm which model is better DONE! 
# 4. SHAPs or coefficient value depending upon 3. DONE! 
# 5. Let's look at if I were to choose the Linear Regression what would 
#     Feature Importance Look like?  

# my recipe went through many iterations
# Iteration #1: this is where I started and trained the GBM on
views_recipe <- recipe(totalViews ~ ., views_data) %>% 
  update_role(id, new_role = "ID") %>% 
  step_rm(creationDate, versionUpdate, title, subtitle, name, id) %>% 
  prep()


# Iteration #2: I wanted to check how the GLM fared.  So I 
#           scaled the predictors between 0 and 1 to make the
#           coeffs comparable
# views_recipe <- recipe(totalViews ~ ., views_data) %>% 
#   update_role(id, new_role = "ID") %>% 
#   step_rm(creationDate, versionUpdate, title, subtitle, name, id) %>% 
#   step_range(all_predictors(), min = 0, max = 1) %>%
#   prep()


# Iteration #3: I had forgotten about "Name" and went back to get the densities
# views_recipe <- recipe(totalViews ~ ., views_data) %>% 
#   update_role(id, new_role = "ID") %>% 
#   step_rm(creationDate, versionUpdate, title, subtitle, id) %>%
#   step_other(name) %>% 
#   step_dummy(name) %>% 
#   step_range(all_predictors(), min = 0, max = 1) %>%
#   prep()


views_final <- bake(views_recipe, new_data = NULL)
final_holdout <- bake(views_recipe, new_data = views_holdout)


views_xgb_spec <- boost_tree(mode = "regression",
           mtry = tune(), 
           min_n = tune(),
           trees = 500,
           learn_rate = 0.3) %>% 
  set_engine("xgboost", objective = "count:poisson") 
  
views_xgb_wf <- workflow() %>% 
  add_formula(totalViews ~ .) %>% 
  add_model(views_xgb_spec)


set.seed(69420)
views_folds <- vfold_cv(data = views_final, v = 4)

doParallel::registerDoParallel(4)
views_grid <- tune_grid(views_xgb_wf,
  resamples = views_folds,
    control = control_grid(save_pred = T,
                           verbose = T))

xgb_results <- views_grid %>% 
  collect_metrics() %>% 
  mutate(model = "xgb")


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

# views_grid <- readRDS(file.path('models', 'xgb_grid.rds'))

best_xgb_spec <- views_grid %>% 
  select_best(metric = "rsq")

# saveRDS(views_grid, file.path('models', 'xgb_grid.rds'))

# FINGERS CROSSED EVERYONE!!!!!!!!
# Okay, I've dropped all 34 of my Mozilla tabs,
# killed my music
# dropped doParallel
# stopped Slack
# FINGERS CROSSED EVERYONE!!!!!!!!
# I'm an idiot...
# lolz
# geez chat what a ride!

best_xgb <- finalize_model(views_xgb_spec, best_xgb_spec) %>% 
  fit(totalViews ~ ., views_final)

saveRDS(best_xgb, file.path('models', 'xgb_model.rds'))


# SHAPS
meme_get("AllTheThings") %>%
  meme_text_bottom("get the SHAPs!")

xgb_preds <- best_xgb %>% 
  predict(final_holdout) %>% 
  bind_cols(final_holdout)

xgb_preds
# those look like reasonable predictions
# Okay, I have something to turn in.... now on to the SHAPs!!!!!

pred_frame_chk <- best_xgb %>% 
  predict(views_final) %>% 
  bind_cols(views_final) %>% 
  mutate(actual_over_expected = totalViews - .pred) 

pred_frame_chk %>% 
  ggplot(aes(x = totalViews, y = .pred)) +
  geom_point() +
  geom_smooth(method = "lm") +
  scale_x_continuous(labels = comma) +
  scale_y_continuous(labels = comma) +
  labs(title = "preds vs. actuals",
       subtitle = "looks pretty good, as it should be since the data was trained on this",
       x = "actuals",
       y = "preds") +
  theme(plot.title.position = "plot")

pred_frame_chk %>% 
  pull(actual_over_expected) %>% 
  hist()

# [1] ".pred"                  "daysSinceCreation"      
# "daysSinceLastUpdate"   # [4] "totalDownloads"         
# "totalVotes"             "totalKernels"     # [7] "datasetTagCount"        "competitionTagCount"    "kernelTagCount"        
# [10] "totalCompressedBytes"   "totalUncompressedBytes" 
# "totalViews"            # [13] "actual_over_expected"


pred_plotter <- function(df, field) {

  ggplot(df, aes(x = {{field}}, y = .pred)) +
    geom_point() +
    geom_smooth() +
    scale_x_continuous(labels = comma) +
    scale_y_continuous(labels = comma) +
    theme(plot.title.position = "plot")
}

pred_plotter(pred_frame_chk)

map(.f = pred_plotter, 
    .x = map(.f = as.name, 
             .x = names(pred_frame_chk)
         )
    , df = pred_frame_chk)

xgb_preds %>% 
  write_csv(file.path(predictions_folder, 'xgb_preds.csv'))

tidy_shaps <- best_xgb %>% 
  predict(final_holdout,
          type = "raw",
          opts = list(predcontrib= T,
                      approxcontrib = F)) %>% 
  as_tibble()

# shap output 
feat_df <- clean_feat_matrix(final_holdout, NULL)
shaps_df <- clean_shaps_matrix(tidy_shaps)
shap_vec <- combine_shaps_and_features(shaps_df, feat_df)

# numeric shap value plots
shapper_numeric(df = shap_vec$numeric) +
  labs(title = "SHAP Values for All Numeric Variables",
       subtitle = "the numeric column is on the x-axis, the SHAP value on the y-axis")

# as Downloads increases, so do views, duh
# as Votes increases, so do views, duh
# competitionTagCount interesting that there's a tail that dips
# dataset Tag Count doesn't seem to matter a whole lot
# kernelTagCount, totalCompressedBytes, totalKernels, meh....
# totalUncompressedBytes moves upwards.  

# altogether alot of stock is being put into the following:
# 1. daysSinceCreation 
# 2. totalDownloads
# 3. totalVotes

# random density plot
shap_vec$numeric %>% 
  ggplot(aes(x = shaps)) +
    geom_density() +
    facet_wrap(~feature, nrow = 5)




######################################## GLM ########################################
################################ Iteration #2 Recipe ################################
views_glm_spec <- poisson_reg(mode = "regression",
    penalty = tune(), 
    mixture = tune()) %>% 
  set_engine("glmnet") 


views_glm_wf <- workflow() %>% 
  add_formula(totalViews ~ .) %>% 
  add_model(views_glm_spec)

views_grid_glm <- tune_grid(views_glm_wf,
  resamples = views_folds,
  control = control_grid(save_pred = T,
                         verbose = T))

# rough go of the cross validation!
views_grid_glm %>% 
  collect_metrics() %>% 
  filter(.metric == "rsq") %>% 
  ggplot(aes(x = penalty, y = mixture, color = mean)) +
  geom_point() +
  scale_color_continuous(type = "viridis") +
  labs(title = "Performance of R^2 on 4 fold cv",
       subtitle = "the lighter the circle, the better the R^2") +
  geom_smooth(method = "lm") +
  theme(plot.title.position = "plot")
# that's one high leverage point out there... this 
# is telling me I have a significatly worse model than 
# the GBM predictions I put up.  
# and it looks like the bigger the penalty the higher 
# the R^2.  slim grid though... 
# let's finalize with the best one so far.  and check the 
# coefficients


glm_results <- views_grid_glm %>% 
  collect_metrics() %>% 
  mutate(model = "glm")


best_glm <- finalize_model(views_glm_spec, best_glm_spec) %>% 
  fit(totalViews ~ ., views_final)

saveRDS(best_glm, file.path('models', 'glm_model.rds'))

# this tells us out best penalty = 0.4
best_glm_spec

# we can toss that 0.4 in here to get the coeffs
tidy_coeffs <- coef(best_glm$fit, s = 0.4) %>% 
  tidy() 

names(tidy_coeffs) <- c("term", "column", "coefficient")

coeff_graph <- tidy_coeffs %>% 
  select(-column ) %>% 
  filter(term != "(Intercept)") %>% 
  mutate(sign = as.factor(sign(coefficient)),
         term = fct_reorder(term, coefficient, .fun = abs)) %>% 
  ggplot(aes(x = term, y = coefficient, color = sign, fill = sign)) + 
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "What are the most important features explaining totalViews?",
       subtitle = "bars are sorted by absolute value of the coefficient from the glm",
       x = NULL,
       y = "coefficient / beta") +
  theme(plot.title.position = "plot",
        legend.position = "none")

gbm_shaps_graph <- shapper_numeric(df = shap_vec$numeric) +
  labs(title = "SHAP Values for All Numeric Variables",
       subtitle = "the numeric column is on the x-axis, the SHAP value on the y-axis")

# Aren't you interest in what the difference is between the GLM and GBM?  
# I know I am... let's bring these together to see what's different / same...
# using patchwork.... 
gbm_shaps_graph + coeff_graph

# I think the most interesting this about the two 
# models is the difference in how they handle daysSinceCreation
# the GBM says that there's a 
# positive relationship between daysSinceCreation and 
# totalViews
#..
#..
#.. 
# that's not the case with the GBM...
# the GBM can pick up that non-linear relationship 
# associated with daysSinceCreation...
# it's able to say a "low" daysSinceCreation value
# negatively affects totalViews and is a deterrent.
# where larger values kind of add to the prediction.  
# pretty cool stuff... 

# okay... I want to make some time series with creation date
# see if there's any trends there that I should 
# have picked up on...

# GLM tuning graph
views_grid_glm %>% 
  collect_metrics() %>% 
  filter(.metric == "rsq") %>% 
  ggplot(aes(x = penalty, y = mixture, color = mean)) +
  geom_point() +
  scale_color_continuous(type = "viridis") +
  labs(title = "Performance of R^2 on 4 fold cv",
       subtitle = "the lighter the circle, the better the R^2") +
  geom_smooth(method = "lm") +
  theme(plot.title.position = "plot")




################################# GBM with Cat Feats ################################
################################ Iteration #3 Recipe ################################
# here I re-trained the gbm above in iteration 1 with name step_other'd, 
# and one hot encoded... 
# hard to believe I had enough time to loop through and CV it
# but honestly I don't remember... 



# new GBM trained on 
feat_df <- clean_feat_matrix(final_holdout, NULL)
shaps_df <- clean_shaps_matrix(tidy_shaps)
shap_vec <- combine_shaps_and_features(shaps_df, feat_df)
shapper_numeric(df = shap_vec$numeric) +
  labs(title = "SHAP Values for All Numeric Variables",
       subtitle = "the numeric column is on the x-axis, the SHAP value on the y-axis")

shapper_cats(df = shap_vec$cats) +
  labs(title = "SHAP Values for All Cat Variables",
       subtitle = "name is grouped on the y, shap density on x")
