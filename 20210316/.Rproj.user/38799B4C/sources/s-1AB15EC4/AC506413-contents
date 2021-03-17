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

games_data <- read_csv(file.path(data_folder, 'sliced_data.csv')) %>%
  mutate(volatile = as.factor(ifelse(volatile %in% c(-1, 1), 1, 0)))
games_holdout <- read_csv(file.path(data_folder, 'sliced_holdout_data.csv'))


# Gameplan:
# 1. transform the target variable
# 1.a. Address the percent format
# 2. speed run a logistic regression model for preds
# 3. speed run to plotting densities of preds
# 4. think about how to bring in the GameName





games_data %>% skim()
# ── Data Summary ────────────────────────
# Values
# Name                       Piped data
# Number of rows             82373
# Number of columns          10
# _______________________
# Column type frequency:
#   character                3
# Date                     1
# numeric                  6
# ________________________
# Group variables            None
#
# ── Variable type: character ────────────────────────────────────────────────────────────────────────────────────────────────
# skim_variable n_missing complete_rate   min   max empty n_unique whitespace
# 1 gamename              0             1     3    81     0     1258          0
# 2 month                 0             1     3     9     0       12          0
# 3 avg_peak_perc         0             1     2     8     0    70498          0
#
# ── Variable type: Date ─────────────────────────────────────────────────────────────────────────────────────────────────────
# skim_variable n_missing complete_rate min        max        median     n_unique
# 1 yearmonth             0             1 2012-08-01 2021-02-01 2018-02-01      103
#
# ── Variable type: numeric ──────────────────────────────────────────────────────────────────────────────────────────────────
# skim_variable n_missing complete_rate       mean        sd       p0    p25     p50    p75     p100 hist
# 1 year                  0             1 2017.          2.22     2012  2016   2018    2019      2021  ▂▃▆▇▅
# 2 avg                   0             1 2746.      26620.          0    53.6  203.    755.  1584887. ▇▁▁▁▁
# 3 gain                  0             1  -10.3      3791.    -250249.  -38.2   -1.62   22.2  426446. ▁▇▁▁▁
# 4 peak                  0             1 5412.      50360.          0   138    498    1703   3236027  ▇▁▁▁▁
# 5 month_num             0             1    6.54        3.52        1     3      7      10        12  ▇▅▅▅▇
# 6 volatile              0             1    0.00883     0.606      -1     0      0       0         1  ▂▁▇▁▂







# DataExplorer::create_report(games_data)

games_recipe <- recipe(volatile ~ ., games_data) %>%
  step_rm(gamename, month, yearmonth) %>% # for now... this is temporary
  step_mutate(avg_peak_perc = as.numeric(stringr::str_replace(avg_peak_perc, pattern = "\\%", "")) / 100) %>%
  prep()


game_training <- bake(games_recipe, new_data = NULL)
final_holdout <- bake(games_recipe, new_data = games_holdout)

game_lr_spec <- parsnip::logistic_reg(
  penalty = 0,
  mixture = 0
) %>%
  set_engine("glmnet")

game_lr <- game_lr_spec %>%
  fit(formula = volatile ~ ., data = game_training)

# Error: cannot allocate vector of size 37.0 Gb
# story of my life!!!!!! ^^^^^^^^^^^^^^^^^^^^^^^

game_lr_preds <- game_lr %>%
  predict(final_holdout)


game_lr_preds_prob <- game_lr %>%
  predict(final_holdout, type = "prob")

preds <- game_lr_preds_prob %>%
  bind_cols(final_holdout) %>%
  bind_cols(game_lr_preds)


#

preds %>%
  write_csv(file.path(preds_folder, 'lm_all_numeric.csv'))

# questions I have about this data....
preds %>% names()

# what is the relationship between all my variables and .pred_1
preds_plotter <- function(var, df) {
  var_name <- as.name(var)

  df %>%
    ggplot(aes(x = !!var_name, y = .pred_1)) +
      geom_point() +
      scale_y_continuous(labels = scales::percent) +
      labs(y = "probability of 'volatility'",
           title = glue::glue('How does {var} relate to volatility?'),
           subtitle = "don't ask about the validity of the smoother... it's staying") +
      geom_smooth()
}


vars_I_care_about <- preds %>%
  select(avg, peak, avg_peak_perc, month_num, year) %>%
  names()

map(vars_I_care_about, preds_plotter, preds)

# gamename : str - video game name
# year : int - year
# month : str - month name
# avg : float - average number of players at the same time
# gain : float - difference in average compared to the previous month (NA = 1st month)
# peak : int - highest number of players at the same time
# avg_peak_perc : str - share of the average in the maximum value (avg / peak) in %
# month_num : int - month in numeric form
# yearmonth : str - date in YYYY-MM-DD format. Note there is no actual day, the date
# formatting defaults to the 1st of the month
# volatile : int - the volatility of users gained based on all other games, with some
# adjustments for time context


# My Model is saying in regards to Skyrim (it's about to be
# incredibly obvious I know nothing about Skyrim if not already known):
# 1. as avg increases -> so does the probability of volatility...
# 2. over time the volatility of Skyrim is dropping. crested in 2015
# 3. seasonality doesn't show too much of an obv pattern
# 4. this one's confusing.... idk about this
# 5. as peak increases, prob ^^^^^^, cappoing around 60000

# no Skyrim in the training data... had to check


# questions about the data I have
# who are the most volatile out of the top 100?
top_50 <- games_data %>%
  count(gamename) %>%
  arrange(desc(n)) %>%
  head(50)

games_top_50 <- games_data %>%
  inner_join(top_50, by = "gamename") %>%
  mutate(volatile = ifelse(volatile == "1", 1, 0)) %>%
  group_by(gamename) %>%
  summarize(perc = sum(volatile) / length(volatile)) %>%
  ungroup() %>%
  mutate(gamename = fct_reorder(gamename, perc))


ggplot(games_top_50, aes(x = gamename, perc)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_y_continuous(labels = scales::percent) +
  labs(x = NULL, y = "volatility as a percent",
      title = "Which games are the most volatile? (top 50)") +
  theme(plot.title.position = "plot")

# I see a lot of first person shooters....
# that's probably not so surprising....
# Age of Empire... hmmm isn't that like Starcraft??? idk
# Batman's volatile?
# interesting...



# games_top_100 <- games_data %>%
#   inner_join(top_100, by = "gamename") %>%
#   mutate(volatile = ifelse(volatile == "1", 1, 0)) %>%
#   filter(gamename == "Age of Empires® III (2007)") %>%
#   group_by(gamename) %>%
#   summarize(perc = sum(volatile) / length(volatile))

#


# next question has to be what does volatility look like over time?
# our model seemed to suggest that there's been
# ____lesss_____ volatility over time, let's compare that to
# what the data is actually saying

# what does overall volatility look like over time?
volatility_over_time <- games_data %>%
  mutate(volatile = ifelse(volatile == "1", 1, 0)) %>%
  group_by(yearmonth) %>%
  summarize(perc = sum(volatile) / length(volatile)) %>%
  ungroup()


ggplot(volatility_over_time, aes(x = yearmonth, y = perc)) +
  geom_line() +
  geom_point() +
  scale_x_date(breaks = "6 months",
               labels = date_format("%b %Y")) +
  scale_y_continuous(labels = scales::percent) +
  labs(x = NULL, y = "volatility as a percent",
       title = "What does overall volatility look like?") +
  geom_smooth() +
  theme(plot.title.position = "plot")
# pretty stable... nothing crazy... I wonder which
# features in our holdout data were making our model
# say that Skyrim has plummetted in volatility...
# this doens't seem to tie to the training data we just looked at


# What about the top 10 games?
# what has their volatility looked like over time?
# did the experience many drops or spikes in volatiltiy?
# I wonder where volatility is the __most__ extreme
# Just something I'm interested in: given a games starting
# date, is volatility more extreme closer to the starting date?
# or, is there a "warm-up" period for the god-tier players
# and once that happens they just _WIPE_ the floor with the
# worst players


# alright, top 10 games most volatile
# as determined by difference between min
# and max volatility

# I want to grab the top 100 to see

# THIS IS TRASH
# top_100  <- games_data %>%
#   count(gamename) %>%
#   arrange(desc(n)) %>%
#   head(100)
#
# max_min_diff <- games_data %>%
#   inner_join(top_100) %>%
#   mutate(volatile = ifelse(volatile == "1", 1, 0)) %>%
#   group_by(gamename, yearmonth) %>%
#   summarize(perc = sum(volatile) / length(volatile)) %>%
#   ungroup() %>%
#   group_by(gamename) %>%
#   summarize(max_perc = max(perc),
#             min_perc = min(perc)) %>%
#   ungroup() %>%
#   mutate(diff = max_perc - min_perc)


# were the top 10 most volatile games
#   _always_ the most volatile?
top20_volatile <- games_data %>%
  mutate(volatile = ifelse(volatile == "1", 1, 0)) %>%
  group_by(gamename) %>%
  summarize(sum_volatile = sum(volatile)) %>%
  ungroup() %>%
  arrange(desc(sum_volatile)) %>%
  head(20)
# for the most part, yes


top20_volatile %>%
  inner_join(games_data) %>%
  mutate(volatile = ifelse(volatile == "1", 1, 0)) %>%
  ggplot(aes(x = yearmonth, y = volatile)) +
    geom_point() +
    geom_line() +
    scale_x_date(breaks = "1 year",
                 labels = date_format("%y")) +
  labs(x = NULL, y = "volatility",
       title = "From the start of our data period, were the top 10 games (by volatility sum) always volatile?",
       subtitle = "Check out: PayDay 2") +
    scale_y_discrete() +
    facet_wrap(~gamename)
# weird slow ramp up with PayDay 2 where it wasn't volatile and then it
# just goes!
# Then there's your 1st person shooters.  Their volatility is almost
# always 1.  Those games probably didn't come out in '13 though.
# the gap between those and PayDay are interesting.

games_data %>%
  mutate(avg_peak_perc = as.numeric(stringr::str_replace(avg_peak_perc, pattern = "\\%", "")) / 100) %>%
  group_by(year(yearmonth), gamename) %>%
  summarize(avg_peak_perc = mean(avg_peak_perc)) %>%
  ggplot(aes(x = `year(yearmonth)`, y = avg_peak_perc, group = `year(yearmonth)`)) +
    geom_boxplot() +
    coord_flip()

# It's like a crappy joy division cover... bleh

games_data %>%
  mutate(avg_peak_perc = as.numeric(stringr::str_replace(avg_peak_perc, pattern = "\\%", "")) / 100) %>%
  ggplot(aes(x = yearmonth, y = avg_peak_perc, group = yearmonth)) +
    ggridges::geom_density_ridges(aes(y = avg_peak_perc)) +
    coord_flip() +
    scale_x_date(breaks = "1 year")
