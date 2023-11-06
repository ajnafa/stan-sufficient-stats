#-----------------------------------------------------------------------------#
#----------Categorical and Multinomial Models for Nominal Outcomes-------------
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

# Function to Simulate the Data for a Categorical Response
sim_multinomial_data <- function(alpha,
                                 beta,
                                 probs = c(0.5, 0.5),
                                 ...) {
    
    # Treatment Assignment
    W = rmultinom(1, 1, prob = probs)
    id = which(W == 1)

    # Potential Outcomes for Each Category
    mu = matrix(0.00, nrow = length(probs), ncol = length(alpha))
    mu[1, ] = alpha
    for (j in 2:nrow(mu)) {
        mu[j, ] = alpha + beta[j - 1, ]
    }

    Mu = matrix(0.00, nrow = length(probs), ncol = length(alpha))
    Y_obs = matrix(0.00, nrow = length(probs), ncol = length(alpha))
    Y_mis = matrix(0.00, nrow = length(probs), ncol = length(alpha))
    for (j in 1:length(alpha)) {
        # Covariance Matrix for the Potential Outcomes
        Sigma_j = diag(length(mu[, j]))

        # Potential Outcomes on the Latent Scale for response category j
        Mu[, j] = MASS::mvrnorm(1, mu[, j], Sigma_j)

        # Observed and Missing Potential Outcomes for response category j
        Y_obs[, j] = Mu[, j] * W + (1 - W) * Mu[, j]
        Y_mis[, j] = Mu[, j] * (1 - W) + W * Mu[, j]
    }

    # Observed Response Category
    theta = exp(Y_obs[id, ]) / sum(exp(Y_obs[id, ]))
    Y = rmultinom(1, 1, theta)
    Y = which(Y == 1)

    # Data for Player i
    profile <- data.table(
        assignment = id,
        outcome = Y,
        truth_mu = Mu[id, Y],
        truth_theta = theta[Y]
    )

    return(profile)
}

# Simulate the Data
plan(multisession, workers = 6)

# Parameters for the Simulated Data for a Categorical Response
n = 500000                          # Number of Observations
alpha = c(0.00, -4.5, -5.5)         # Intercept for Each Category
probs = c(1/3, 1/3, 1/3)            # Treatment Assignment Probabilities
beta = matrix(                      # Treatment Effects for Each Category and Condition
    c(0.00, 0.8, -1.0,
      0.00, -0.8, 0.5),
    nrow = 2,
    ncol = 3,
    byrow = TRUE
)

# Simulate the data for each player
sim_df <- future_map(
    .x = 1:n,
    .f = ~ sim_multinomial_data(
        alpha = alpha,
        beta = beta,
        probs = probs
    ),
    .progress = TRUE,
    .options = furrr_options(seed = TRUE)
)

# Bind the data together
df <- rbindlist(sim_df, idcol = 'obs_id')
df[, treat := factor(
    assignment, 
    levels = 1:3, 
    labels = c("Control", "Treatment 1", "Treatment 2")
)]

#-----------------------------------------------------------------------------#
#--------Categorical and Multinomial Models for Non-Hierarchical Data----------
#-----------------------------------------------------------------------------#

# Load the Stan Models
catlogit_stan <- cmdstan_model("models/multinomial/categorical-logit.stan")

# Prepare the data for the categorical model
X_cat <- model.matrix(~ treat, data = df)
stan_data_cat <- list(
    N = nrow(df),
    D = ncol(X_cat),
    K = max(df$outcome),
    Y = df$outcome,
    P = X_cat
)

# Fit the Categorical Logit Model
catlogit_fit <- catlogit_stan$sample(
    data = stan_data_cat,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 500,
    iter_sampling = 500,
    save_warmup = FALSE,
    refresh = 25
)

# Summarize the results
catlogit_fit$summary()

# Load the Stan Models
multilogit_stan <- cmdstan_model("models/multinomial/multinomial-logit.stan")

# Prepare the data for the Multinomial model
stan_data_mlogit <- list(
    N = nrow(df),
    D = max(df$assignment),
    K = max(df$outcome),
    Y = df$outcome,
    dd = df$assignment
)

# Fit the Multinomial Logit Model
multilogit_fit <- multilogit_stan$sample(
    data = stan_data_mlogit,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 500,
    iter_sampling = 500,
    save_warmup = FALSE,
    refresh = 25
)

# Summarize the results
multilogit_fit$summary()
