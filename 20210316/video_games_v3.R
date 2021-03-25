source('utils.R')


games_data <- read_csv(file.path(data_folder, 'sliced_data.csv')) %>%
  mutate(volatile = as.factor(ifelse(volatile %in% c(-1, 1), 1, 0))) %>%
  janitor::clean_names("small_camel")
games_holdout <- read_csv(file.path(data_folder, 'sliced_holdout_data.csv')) %>%
  janitor::clean_names("small_camel")


games_recipe <- recipe(volatile ~ ., games_data) %>%
  step_rm(gamename, month, yearmonth) %>% # for now... this is temporary
  step_mutate(avgPeakPerc = as.numeric(stringr::str_replace(avgPeakPerc, pattern = "\\%", "")) / 100) %>%
  step_meanimpute(avgPeakPerc) %>%
  step_range(all_predictors()) %>%
  prep()


game_training <- bake(games_recipe, new_data = NULL)
final_holdout <- bake(games_recipe, new_data = games_holdout)


video_game_lr_spec <- logistic_reg(
  mode = "classification",
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet")

set.seed(69420)
vg_folds_2 <- vfold_cv(data = game_training, v = 4)

vg_workflow_lr <- workflow() %>%
  add_formula(volatile ~ .) %>%
  add_model(video_game_lr_spec)

doParallel::registerDoParallel(4)
vg_grid_lr <- tune_grid(
  object = vg_workflow_lr,
  resamples = vg_folds_2,
  control = control_grid(verbose = T,
                         save_pred = T),
  metrics = metric_set(roc_auc, pr_auc, accuracy),
  grid = 10
)

vg_metrics <- vg_grid_lr %>%
  collect_metrics()

vg_grid_lr %>% saveRDS(file.path(models_folder, 'lr_grid.rds'))

best_lr <- vg_grid_lr %>%
  select_best("accuracy")


video_game_lr <- video_game_lr_spec %>%
  finalize_model(best_lr) %>%
  fit(volatile ~ ., game_training)

video_game_lr %>% saveRDS(file.path(models_folder, 'lr_model.rds'))

lr_preds <- video_game_lr %>%
  predict(final_holdout, "prob") %>%
  bind_cols(final_holdout)

video_game_lr %>%
  tidy() %>%
  filter(term != "(Intercept)") %>%
  mutate(term = fct_reorder(term, estimate, abs)) %>%
  ggplot(aes(x = term, y = estimate, color = as.factor(sign(estimate)), fill = as.factor(sign(estimate)))) +
    geom_col(stat = "identity") +
    coord_flip() +
    labs(y = "coefficient",
         x = "feature",
         title = "Feature Importance",
         subtitle = "sorted by absolute value of coefficient"
         ) +
  theme(plot.title.position = "plot",
        legend.position = "none")

probability_mapper <- function(x) {
  x_name <- as.name(x)

  lr_preds %>%
    ggplot(aes(x = !!x_name, y = .pred_1)) +
      geom_point() +
      geom_smooth() +
      labs(title = glue::glue("How volatility relates to {x_name}"),
          y = "Probability of Volatility",
          x = NULL)
}


map(names(lr_preds)[3:8], .f = probability_mapper)




