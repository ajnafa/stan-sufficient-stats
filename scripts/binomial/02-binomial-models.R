#-----------------------------------------------------------------------------#
#-------------Bernoulli and Binomial Models for Binary Outcomes----------------
#-Author: A. Jordan Nafa-----------------------------------------License: MIT-#

# Load required libraries
pacman::p_load(
    data.table,
    arrow,
    cmdstanr,
    posterior,
    install = FALSE
)

#------------------------------------------------------------------------------#
#----------------------------Bernoulli Model(s)---------------------------------
#------------------------------------------------------------------------------#

# Load the Stan Models
bernoulli_stan <- cmdstan_model("models/binomial/bernoulli-logit.stan")

# Load the Data
df <- read_parquet("output/data/bernoulli_sim_data.parquet")

# Prepare the data for the Bernoulli Model
X_bernoulli <- model.matrix(~ treat + age, data = df)
stan_data_bernoulli <- list(
    N = nrow(df),
    D = ncol(X_bernoulli),
    Y = df$outcome_obs,
    P = X_bernoulli,
    mu_alpha = -3.5,
    sigma_alpha = 1,
    mu_beta = c(0, 0, 0),
    sigma_beta = c(0.75, 0.75, 0.75),
    treat_idx = 2
)

# Fit the Bernoulli Model
bernoulli_fit <- bernoulli_stan$sample(
    data = stan_data_bernoulli,
    output_dir = "output/binomial/bernoulli-logit",
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1000,
    save_warmup = FALSE,
    refresh = 25
)
 
# Save the Bernoulli Model
bernoulli_fit$save_object(
    "output/binomial/bernoulli-logit/bernoulli_fit.Rds"
)

# Extract the Parameter Estimates
bernoulli_draws <- bernoulli_fit$draws(format = 'draws_df')

# Summarize the Parameter Estimates
bernoulli_draws_summ <- summarise_draws(bernoulli_draws)
print(bernoulli_draws_summ)

#------------------------------------------------------------------------------#
#-----------------------------Binomial Model(s)---------------------------------
#------------------------------------------------------------------------------#

# Aggregate the data for the Binomial Model
df_agg = df[, .(
    n = .N,
    outcome_obs = sum(outcome_obs),
    outcome_mis = sum(outcome_mis)
), by = .(treat, assignment, subgroup, age)]
setorder(df_agg, assignment, subgroup)

# Load the Binomial Model
binomial_stan <- cmdstan_model("models/binomial/binomial-logit.stan")

# Prepare the data for the Binomial Model
X_binomial <- model.matrix(~ treat + age, data = df_agg)
stan_data_binomial <- list(
    N = nrow(df_agg),
    D = ncol(X_binomial),
    Y = df_agg$outcome_obs,
    T = df_agg$n,
    P = X_binomial,
    mu_alpha = -3.5,
    sigma_alpha = 1,
    mu_beta = c(0, 0, 0),
    sigma_beta = c(0.75, 0.75, 0.75),
    treat_idx = 2
)

# Fit the Binomial Model
binomial_fit <- binomial_stan$sample(
    data = stan_data_binomial,
    output_dir = "output/binomial/binomial-logit",
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1000,
    save_warmup = FALSE,
    refresh = 25
)

# Save the Binomial Model
binomial_fit$save_object(
    "output/binomial/binomial-logit/binomial_fit.Rds"
)

# Extract the Parameter Estimates
binomial_draws <- binomial_fit$draws(format = 'draws_df')

# Summarize the Parameter Estimates
binomial_draws_summ <- summarise_draws(binomial_draws)
print(binomial_draws_summ)

# Make a named list of the parameter summaries
parameter_summaries <- list(
    bernoulli = bernoulli_draws_summ,
    binomial = binomial_draws_summ
)

# Bind the parameter summaries together
rbindlist(parameter_summaries, idcol = 'model')

