library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(skimr)
library(ggridges)

# EVERY DFKDSFD:SKF TIME !!!!!
theme_set(theme_minimal())

data_folder <- 'data'
preds_folder <- 'preds'
plots_folder <- 'plots'
models_folder <- 'models'


clean_feat_matrix <- function(df_before_dmatrix, response_variable) {
  #' @param: df_before_dmatrix {tibble} the tibble before it's turned into a dmatrix
  #' @param: response_variable {string} the response variable to be predicted
  #'

  if (!is.null(response_variable)) {
    response_name <- as.name(response_variable)

    long_feat_df <- df_before_dmatrix %>%
      mutate(id = row_number()) %>%
      pivot_longer(names_to = "features",
                   values_to = "feature_value",
                   cols = -c(id, all_of(response_name)))
  } else {
    long_feat_df <- df_before_dmatrix %>%
      mutate(id = row_number()) %>%
      pivot_longer(names_to = "features",
                   values_to = "feature_value",
                   cols = -c(id))
  }

  long_feat_df <-  long_feat_df %>%
    filter(feature_value != 0) %>%
    mutate(features = str_replace_all(features, pattern = "\\.+", replacement = '_')) %>%
    separate(features,
             into = c("feature", "subfeature"),
             remove = FALSE,
             extra = "merge")

  long_feat_df
}

clean_shaps_matrix <- function(xgb_preds) {
  #' @param: xgb_preds {tibble} the tibble after predicting SHAPs from tidy models shap
  #'

  shaps_df <- as.data.frame(xgb_preds) %>%
    as_tibble() %>%
    select(-BIAS) %>%
    mutate(id = row_number()) %>%
    pivot_longer(names_to = "subfeatures", values_to = "shaps",
                 cols = -id) %>%
    mutate(subfeatures = str_replace_all(subfeatures,
                                         pattern = "\\.+",
                                         replacement = '_')) %>%
    separate(subfeatures,
             into = c("feature", "subfeature"),
             remove = FALSE,
             extra = "merge") %>%
    group_by(id, feature) %>%
    summarize(shaps = sum(shaps), .groups = "drop")
}

combine_shaps_and_features <- function(shaps_frame, feat_df) {
  #' @param: shap_frame {tibble}: the tibble from `clean_shaps_matrix`
  #' @param: feat_df {tibble}: the tibble from `clean_feat_matrix`

  shaps_to_graph <- shaps_frame %>%
    inner_join(feat_df, by = c("id", "feature"))

  cat_feats <- shaps_to_graph %>%
    filter(!is.na(subfeature))

  num_feats <- shaps_to_graph %>%
    filter(is.na(subfeature))

  return_vec <- c()
  return_vec[['cats']] <- cat_feats
  return_vec[['numeric']] <- num_feats

  return_vec
}

shapper_cats <- function(x = NULL, df, sample_size = NULL) {
  #' @param: x {string}: the categorical variable to plot grouped densities from
  #' @param: df {tibble}: the categorical data frame output from `combine_shaps_and_features`
  #' @param: sample_size {int}: how many shap value to sample from the total
  #'

  if (!is.null(sample_size)) {
    set.seed(69420)
    df <- df %>%
      sample_n(sample_size)
  }

  if (!is.null(x)) {
    df <- df %>%
      filter(str_detect(string = feature, pattern = x)) %>%
      mutate(subfeature = fct_reorder(subfeature, shaps, .fun = median))

    var_for_title <- unique(df$feature)[1]
  } else {
    df <- df %>%
      mutate(subfeature = fct_reorder(subfeature, shaps, .fun = median))

    var_for_title <- "All Categorical Variables"
  }

  df %>%
    ggplot(aes(x = shaps, y = subfeature, alpha = 0.7)) +
    ggridges::geom_density_ridges() +
    stat_summary(fun = "median", geom = "errorbar", aes(xmax = ..x.., xmin = ..x..), linetype = 2) +
    labs(x = NULL,
         y = NULL,
         title = glue::glue("SHAP Values for {var_for_title}"),
         subtitle = 'sorted by median SHAP value, within group median listed with dotted line') +
    theme(legend.position = "none",
          plot.title.position = "plot")
}

shapper_numeric <- function(x = NULL, df, sample_size = NULL) {
  #' @param: x {string}: the numerical variable to plot densities from
  #' @param: df {tibble}: the numeric data frame output from `combine_shaps_and_features`
  #' @param: sample_size {int}: how many shap value to sample from the total
  #'


  if (!is.null(sample_size)) {
    set.seed(69420)
    df <- df %>%
      sample_n(sample_size)
  }

  if (!is.null(x)) {
    df <- df %>%
      filter(str_detect(string = feature, pattern = x))

    var_for_title <- unique(df$feature)[1]
  } else {
    var_for_title <- "All Numeric Variables"
  }

  df %>%
    ggplot(aes(x = feature_value, y = shaps)) +
    geom_point() +
    geom_smooth(method = "lm") +
    facet_wrap(~feature, scales = "free_x") +
    labs(x = NULL,
         y = NULL,
         title = glue::glue("SHAP Values for {var_for_title}")) +
    theme(legend.position = "none",
          plot.title.position = "plot")
}
