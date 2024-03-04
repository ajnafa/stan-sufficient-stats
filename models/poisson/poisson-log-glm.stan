/* 
    Model: Poisson Likelihood with a Log Link
    Author: A. Jordan Nafa
    Date: 2024-03-03
    License: MIT
*/

data {
    // Data Dimensions
    int<lower=0> N;                             // N observations
    int<lower=1> D;                             // D Predictors

    // Input Data
    array[N] int<lower=0> Y;                    // Response Array
    matrix[N,D] P;                              // Design Matrix

    // Priors for the Parameters
    real mu_alpha;                              // Intercept Prior Mean
    real<lower=0> sigma_alpha;                  // Intercept Prior Std Dev
    vector[D-1] mu_beta;                        // Coefficients Prior Mean
    vector<lower=0>[D-1] sigma_beta;            // Coefficients Prior Std Dev

    // Additional Arguments
    int<lower=2, upper=D> treat_idx;            // Index of the Treatment Variable
}

transformed data {
    int K = D - 1;                              // Number of Coefficients
    matrix[N, K] X;                             // Design Matrix

    // Design Matrix for the Coefficients
    X = P[, 2:D];

    // Index of the Treatment Variable in X
    int idx = treat_idx - 1;

    // Counteractual Matrices for the Treatment Effects
    matrix[N, K] X0 = X;                        // Contrasts Under Y(0, Z)
    X0[, idx] = zeros_vector(N);

    matrix[N, K] X1 = X;                        // Contrasts Under Y(1, Z)
    X1[, idx] = ones_vector(N);
}

parameters {
    real alpha;                                 // Intercept
    vector[K] beta;                             // Coefficients
}

model {
    // Likelihood, Log Link GLM
    profile("Likelihood") {
        target += poisson_log_glm_lpmf(Y | X, alpha, beta);
    }

    // Priors
    profile("Priors") {
        target += normal_lpdf(alpha | mu_alpha, sigma_alpha);
        target += normal_lpdf(beta | mu_beta, sigma_beta);
    }
}

generated quantities {
    // Predictive Distribution of the AME
    real AME;                            // Average Marginal Effect
    real EY1;                            // Expected Value of Y(1, Z)
    real EY0;                            // Expected Value of Y(0, Z)
    {
        // Sample from the Posterior Predictive Distribution
        array[N] real Y1 = poisson_log_rng(alpha + X0 * beta);
        array[N] real Y0 = poisson_log_rng(alpha + X1 * beta);

        // Expected Values of Y(1, Z) and Y(0, Z)
        EY1 = mean(Y1);
        EY0 = mean(Y0);

        // Average Marginal Effect
        AME = EY1 - EY0;
    }
}
