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

# Function to Simulate the Data for a Bernoulli Response
sim_data <- function(alpha,
                     beta,
                     sigma,
                     probs,
                     n_groups = NULL,
                     group_effects = NULL,
                     group_sd = NULL,
                     hierarchical = FALSE,
                     ...) {

    # Treatment Assignment
    W = rmultinom(1, 1, probs)
    id = which(W == 1)

    if (isTRUE(hierarchical)) {
        G = sample(1:n_groups, 1)
        tau = rhnorm(1, group_sd)
        alpha = alpha + group_effects[G] * tau
    } else {
        G = NULL
    }

    # Linear Predictor(s)
    mu = rep(NA_real_, length(W))
    mu[1] = alpha;
    for (j in 2:length(mu)) {
        mu[j] = alpha + beta[j-1]
    }

    # Covariance Matrix for the Potential Outcomes
    Sigma = diag(length(mu))

    # Potential Outcomes on the Latent Scale
    Mu = MASS::mvrnorm(1, mu, Sigma)

    # Latent Observed and Missing Potential Outcomes
    Y_obs = Mu * W + (1 - W) * Mu
    Y_mis = Mu * (1 - W) + W * Mu
    
    # Observed Outcome
    theta = exp(Y_obs) / (1 + exp(Y_obs))
    Y = rbinom(1, 1, theta[id])

    # Data for the ith Observation
    profile <- data.table(
        assignment = id,
        outcome = Y,
        group = G,
        truth_mu = mu[id],
        truth_theta = exp(mu[id]) / (1 + exp(mu[id]))
    )

    return(profile)
}

#-----------------------------------------------------------------------------#
#------------------Simulate Data for the Hierarchical Models-------------------
#-----------------------------------------------------------------------------#

# Parameters for the Simulated Data
n = 100000                              # Number of observations
alpha = -4.0                            # Intercept on the Logit Scale
beta = c(0.8, -1.0)                     # Treatment Effects on the Logit Scale
sigma = c(1.0, 1.0, 1.0)                # Standard Dev. of the Potential Outcomes
probs = c(1/3, 1/3, 1/3)                # Treatment Assignment Probabilities
n_groups = 25                           # Number of Groups
group_effects = rnorm(n_groups, 0, 1.5) # Group Effects
group_sd = 1.5                          # Standard Deviation of the Group Effects

# Planning for the future
plan(multisession, workers = 6)

# Simulate the data for each player
sim_df <- future_map(
    .x = 1:n,
    .f = ~ sim_data(
        alpha = alpha,
        beta = beta,
        sigma = sigma,
        probs = probs,
        n_groups = n_groups,
        group_effects = group_effects,
        group_sd = group_sd,
        hierarchical = TRUE
    ),
    .progress = TRUE,
    .options = furrr_options(seed = TRUE)
)

# Back from the future
plan(sequential)

# Bind the data together
df <- rbindlist(sim_df, idcol = 'obs_id')
df[, treat := factor(
    assignment, 
    levels = 1:3, 
    labels = c("Control", "Treatment 1", "Treatment 2")
)]

setorder(df, group, assignment)

# Write the data to disk
write_parquet(df, "output/data/hierarchical_bernoulli_sim_data.parquet")

# Aggregate the data for the Binomial Model
df_agg <- df[, .(
    n = .N,
    y = sum(outcome),
    mu = mean(truth_mu),
    theta = mean(truth_theta)
), by = .(group, treat, assignment)]

# Write the data to disk
write_parquet(df_agg, "output/data/hierarchical_binomial_sim_data.parquet")

#-----------------------------------------------------------------------------#
#------------Bernoulli and Binomial Models for Hierarchical Data---------------
#-----------------------------------------------------------------------------#

# Load the Stan Models
hierachical_bernoulli_stan <- cmdstan_model(
    "models/binomial/bernoulli-logit-hierarchical.stan"
)

# Prepare the data for the Bernoulli Model
X_hbernoulli <- model.matrix(~ treat, data = df)
stan_data_hbernoulli <- list(
    N = nrow(df),
    D = ncol(X_hbernoulli),
    Y = df$outcome,
    P = X_hbernoulli,
    J = n_groups,
    jj = df$group,
    mu_alpha = -3.5,
    sigma_alpha = 1,
    mu_beta = 0,
    sigma_beta = 0.75,
    sigma_upsilon = 1
)

# Fit the Hierarchical Bernoulli Model
hier_bernoulli_fit <- hierachical_bernoulli_stan$sample(
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
    variables = c("alpha", "beta", "upsilon"), 
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
