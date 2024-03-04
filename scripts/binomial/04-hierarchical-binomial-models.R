#-----------------------------------------------------------------------------#
#-------Hierarchical Bernoulli and Binomial Models for Binary Outcomes---------
#-Author: A. Jordan Nafa-----------------------------------------License: MIT-#

# Load required libraries
pacman::p_load(
    data.table,
    furrr,
    arrow,
    extraDistr,
    cmdstanr,
    posterior,
    install = FALSE
)

# Load the Stan Models
hbernoulli_stan <- cmdstan_model("models/binomial/bernoulli-logit-hierarchical.stan")

# Prepare the data for the Bernoulli Model
X_hbernoulli <- model.matrix(~ treat + age, data = df)
stan_data_hbernoulli <- list(
    N = nrow(df),
    D = ncol(X_hbernoulli),
    Y = df$outcome_obs,
    P = X_hbernoulli,
    J = length(unique(df$group)),
    jj = df$group,
    mu_alpha = -3.5,
    sigma_alpha = 1,
    mu_beta = c(0, 0, 0),
    sigma_beta = c(0.75, 0.75, 0.75),
    treat_idx = 2,
    sigma_upsilon = 1,
    G = 1
)

# Fit the Hierarchical Bernoulli Model
hier_bernoulli_fit <- hbernoulli_stan$sample(
    data = stan_data_hbernoulli,
    output_dir = "output/binomial/bernoulli-logit",
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1000,
    save_warmup = FALSE,
    refresh = 25
)

# Save the Bernoulli Model
hier_bernoulli_fit$save_object(
    "output/binomial/bernoulli-logit/hierarchical-bernoulli_fit.Rds"
)

# Extract the Parameter Estimates
hier_bernoulli_draws <- hier_bernoulli_fit$draws(
    variables = c("alpha", "beta", "EY1", "EY0", "AME"), 
    format = 'draws_df'
)

# Summarize the Parameter Estimates
hier_bernoulli_draws_summ <- summarise_draws(hier_bernoulli_draws)

# Load the Binomial Model
hierarchical_binomial_stan <- cmdstan_model(
    "models/binomial/binomial-logit-hierarchical.stan"
)

# Prepare the data for the Binomial Model
X_hbinomial <- model.matrix(~ treat, data = df_agg)
stan_data_hbinomial <- list(
    N = nrow(df_agg),
    D = ncol(X_hbinomial),
    Y = df_agg$y,
    K = df_agg$n,
    P = X_hbinomial,
    J = n_groups,
    jj = df_agg$group,
    mu_alpha = -3.5,
    sigma_alpha = 1,
    mu_beta = 0,
    sigma_beta = 0.75,
    sigma_upsilon = 1
)

# Fit the Binomial Model
hier_binomial_fit <- hierarchical_binomial_stan$sample(
    data = stan_data_hbinomial,
    output_dir = "output/binomial/binomial-logit",
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1000,
    save_warmup = FALSE,
    refresh = 25
)

# Save the Binomial Model
hier_binomial_fit$save_object(
    "output/binomial/binomial-logit/binomial-logit-hierarchical.Rds"
)

# Extract the Parameter Estimates
hier_binomial_draws <- hier_binomial_fit$draws(
    variables = c("alpha", "beta", "upsilon"), 
    format = 'draws_df'
)

# Summarize the Parameter Estimates
hier_binomial_draws_summ <- summarise_draws(hier_binomial_draws)

# Make a named list of the parameter summaries
parameter_summaries <- list(
    hier_bernoulli = hier_bernoulli_draws_summ,
    hier_binomial = hier_binomial_draws_summ
)

# Bind the parameter summaries together
rbindlist(parameter_summaries, idcol = 'model')
