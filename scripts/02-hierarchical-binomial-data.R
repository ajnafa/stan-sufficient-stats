#-----------------------------------------------------------------------------#
#---Data for Hierarchical Bernoulli and Binomial Models for Binary Outcomes----
#-Author: A. Jordan Nafa-----------------------------------------License: MIT-#

# Load required libraries
pacman::p_load(
    data.table,
    furrr,
    arrow,
    extraDistr,
    MASS,
    install = FALSE
)

# Seed for Reproducibility
set.seed(123)

# Parameters for the Simulated Data
n = 250000                              # Number of observations
alpha = -4.0                            # Intercept on the Logit Scale
beta = 0.8                              # Treatment Effect on the Logit Scale
sigma = c(1.0, 1.0)                     # Standard Dev. of the Potential Outcomes
prob = 0.5                              # Treatment Assignment Probability
rho = 0.00                              # Correlation between the Potential Outcomes
z_probs = c(2/4, 1/4, 1/4)              # Subgroup Probabilities
gamma = c(0.0, 0.5, -0.5)               # Group-specific baselines
n_groups = 25                           # Number of Groups
group_effects = rnorm(n_groups, 0, 1.5) # Group Effects
group_sd = 1.5                          # Standard Deviation of the Group Effects

# Simulate the Treatment Assignment Mechanism
X <- rbinom(n, 1, prob)

# Covariate for the Subgroup Membership
Z <- sample(1:length(z_probs), n, replace = TRUE, prob = z_probs)

# Simulate the Group Effects
G = sample(1:n_groups, n, replace = TRUE)
tau = rhnorm(n, group_sd)
upsilon = group_effects[G] * tau

# Linear Predictor for the Potential Outcomes
mu0 = alpha + beta * 0
mu1 = alpha + beta * 1

# Covariance Matrix for the Potential Outcomes
Sigma = matrix(
    c(sigma[1]^2, rho * sigma[1] * sigma[2], 
    rho * sigma[1] * sigma[2], sigma[2]^2),
    nrow = 2
)

# Subgroup Coefficients
gammaZ = gamma[Z] + rnorm(n, 0, 0.2)

# Simulate the Potential Outcomes 
Mu = mvrnorm(n, mu = c(mu0, mu1), Sigma = Sigma)
Y0 = Mu[, 1]                        # Potential Outcome under Y(0)
Y1 = Mu[, 2]                        # Potential Outcome under Y(1)

# Simulate the Observed and Missing Outcomes On the Logit Scale
y_obs = Y0 * (1 - X) + Y1 * X + gammaZ + upsilon
y_mis = Y0 * X + Y1 * (1 - X) + gammaZ + upsilon

# Simulate the Observed and Missing Outcomes on the Response Scale
y_obs = rbinom(n, 1, plogis(y_obs))
y_mis = rbinom(n, 1, plogis(y_mis))

# Create the Data Frame
df = data.table(
    obs_id = 1:n,
    assignment = X,
    outcome_obs = y_obs,
    outcome_mis = y_mis,
    subgroup = Z,
    group = G
)

# Label the Treatment Assignment
df[, `:=` (
    treat = factor(
        assignment, 
        levels = 0:1, 
        labels = c("Control", "Treatment")
        ),
    age = factor(
        subgroup, 
        levels = 1:3, 
        labels = c("18-35", "36-64", "65+")
        )
    )
]

# Write the data to disk
write_parquet(df, "output/data/hierarchical_bernoulli_sim_data.parquet")
