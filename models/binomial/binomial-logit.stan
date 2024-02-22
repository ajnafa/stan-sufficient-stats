/* 
    Model: Binomial Likelihood with a Logit Link, Sufficient
           Formulation of the Bernoulli Likelihood
    Author: A. Jordan Nafa
    Date: 2024-02-21
    License: MIT
*/

data {
    // Data Dimensions
    int<lower=0> N;                             // N Cells
    int<lower=1> D;                             // D Predictors

    // Input Data
    array[N] int<lower=0> Y;                    // Observed Successes
    array[N] int<lower=0> T;                    // Observed Trials
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
    // Linear Predictor on the Logit Scale
    vector[N] mu = alpha + X * beta;

    // Binomial Likelihood
    profile("Likelihood") {
        target += binomial_logit_lpmf(Y | T, mu);
    }

    // Priors
    profile("Priors") {
        target += normal_lpdf(alpha | mu_alpha, sigma_alpha);
        target += normal_lpdf(beta | mu_beta, sigma_beta);
    }
}

generated quantities {
    real AME;                            // Average Marginal Effect
    real EY1;                            // Expected Value of Y(1, Z)
    real EY0;                            // Expected Value of Y(0, Z)

    /* Calculate Quantities in curly brackets so we aren't writing 
    things we don't need to the CSV files */
    {
        
        // Counterfactual Draws from the Posterior Predictive Distribution
        array[N] real Y0 = binomial_rng(T, inv_logit(alpha + X0 * beta));
        array[N] real Y1 = binomial_rng(T, inv_logit(alpha + X1 * beta));

        // Counterfactual Probabilities
        vector[N] P0 = to_vector(Y0) ./ to_vector(T);
        vector[N] P1 = to_vector(Y1) ./ to_vector(T);

        // Quantities of Interest
        EY0 = mean(P0);
        EY1 = mean(P1);
        AME = EY1 - EY0;
    }
}