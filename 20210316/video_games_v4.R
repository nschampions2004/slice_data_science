source('utils.R')


games_data <- read_csv(file.path(data_folder, 'sliced_data.csv')) %>%
  mutate(volatile = as.factor(volatile)) %>%
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
  metrics = metric_set(accuracy),
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


# comparing models
mult_reg_grid <- readRDS(file.path(models_folder, "mr_grid.rds"))
best_mult_model <- readRDS(file.path(models_folder, 'mr_model.rds'))

mult_reg_col_add <- mult_reg_grid %>%
  collect_metrics() %>%
  mutate(model = "multinomial_regression")

gbm_metrics <- vg_metrics %>%
  mutate(model = "xgboost")


mult_reg_col_add %>%
  bind_rows(gbm_metrics) %>%
  filter(.metric == "accuracy") %>%
  arrange(desc(mean))



tidy(best_mult_model) %>%
  filter(term != "(Intercept)") %>%
  group_by(class) %>%
  mutate(term = fct_reorder(term, estimate, abs)) %>%
  ungroup() %>%
  ggplot(aes(x = term, y = estimate, color = as.factor(sign(estimate)), fill = as.factor(sign(estimate)))) +
    geom_bar(stat = "identity") +
    facet_wrap(~class) +
    coord_flip() +
    scale_y_continuous(labels = scales::comma) +
    labs(title = "Feature Importance for the Multiclass Model",
         y = NULL,
         x = NULL) +
    theme(plot.title.position = "plot",
          legend.position = "none")





plot_fi_bars <- function(x) {
  browser()
  tidy(best_mult_model) %>%
    filter(term == !!x ) %>%
    group_by(class) %>%
    mutate(term = fct_reorder(term, estimate, abs)) %>%
    ungroup() %>%
    ggplot(aes(x = term, y = estimate, color = as.factor(sign(estimate)), fill = as.factor(sign(estimate)))) +
    geom_bar(stat = "identity") +
    facet_wrap(~class) +
    coord_flip() +
    scale_y_continuous(labels = scales::comma) +
    labs(title = "Feature Importance for the Multiclass Model",
         y = NULL,
         x = NULL) +
    theme(plot.title.position = "plot",
          legend.position = "none")
}


names_to_graph <- tidy(best_mult_model) %>%
  pull(term) %>%
  unique() %>%
  map(.f = as.name)

map(.x = names_to_graph, .f = plot_fi_bars)


