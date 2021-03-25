source('utils.R')


games_data <- read_csv(file.path(data_folder, 'sliced_data.csv')) %>%
  mutate(volatile = as.factor(ifelse(volatile %in% c(-1, 1), 1, 0))) %>%
  janitor::clean_names("small_camel")
games_holdout <- read_csv(file.path(data_folder, 'sliced_holdout_data.csv')) %>%
  janitor::clean_names("small_camel")


games_recipe <- recipe(volatile ~ ., games_data) %>%
  step_rm(gamename, month, yearmonth) %>% # for now... this is temporary
  step_mutate(avgPeakPerc = as.numeric(stringr::str_replace(avgPeakPerc, pattern = "\\%", "")) / 100) %>%
  prep()


game_training <- bake(games_recipe, new_data = NULL)
final_holdout <- bake(games_recipe, new_data = games_holdout)


video_game_gbm_spec <- boost_tree(
  mode = "classification",
  trees = 1000,
  mtry = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost")

set.seed(69420)
vg_folds <- vfold_cv(data = game_training, v = 4)

vg_workflow <- workflow() %>%
  add_formula(volatile ~ .) %>%
  add_model(video_game_gbm_spec)

doParallel::registerDoParallel(8)
vg_grid <- tune_grid(
  object = vg_workflow,
  resamples = vg_folds,
  control = control_grid(verbose = T,
                         save_pred = T),
  metrics = metric_set(roc_auc, pr_auc, accuracy),
  grid = 10
)

vg_metrics <- vg_grid %>%
  collect_metrics()

best_gbm <- vg_grid %>%
  select_best('accuracy')

video_game_gbm <- video_game_gbm_spec %>%
  finalize_model(best_gbm) %>%
  fit(volatile ~ ., game_training)


gbm_preds <- video_game_gbm %>%
  predict(final_holdout) %>%
  bind_cols(final_holdout)


tidy_shaps <- video_game_gbm %>%
  predict(final_holdout,
          type = "raw",
          opts = list(predcontrib = T,
                      approxcontrib = F)) %>%
  as_tibble()


# shap output
feat_df <- clean_feat_matrix(final_holdout, NULL)
shaps_df <- clean_shaps_matrix(tidy_shaps)
shap_vec <- combine_shaps_and_features(shaps_df, feat_df)
shapper_numeric(df = shap_vec$numeric)



# comparing models
log_reg_grid <- readRDS(file.path(models_folder, "lr_grid.rds"))
best_lr_model <- readRDS(file.path(models_folder, 'lr_model.rds'))

log_reg_col_add <- log_reg_grid %>%
  collect_metrics() %>%
  mutate(model = "logistic_regression")

gbm_metrics <- vg_metrics %>%
  mutate(model = "xgboost")


log_reg_col_add %>%
  bind_rows(gbm_metrics) %>%
  filter(.metric == "accuracy") %>%
  arrange(desc(mean))
