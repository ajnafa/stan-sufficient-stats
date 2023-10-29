/* 
    Model: Bernoulli Likelihood with a Logit Link
    Author: A. Jordan Nafa
    Date: 2023-10-29
    License: MIT
*/

data {
    // Data Dimensions
    int<lower=0> N;                             // N observations
    int<lower=1> D;                             // K Predictors

    // Input Data
    array[N] int<lower=0,upper=1> Y;            // Binary Response 
    matrix[N,D] P;                              // Design Matrix

    // Priors for the Parameters
    real mu_alpha;                              // Intercept Prior Mean
    real<lower=0> sigma_alpha;                  // Intercept Prior Std Dev
    vector[D-1] mu_beta;                        // Coefficients Prior Mean
    real<lower=0> sigma_beta;                   // Coefficients Prior Std Dev

    // Prior Predictive Check Flag
    int<lower=0,upper=1> prior_check;
}

transformed data {
    int K = D - 1;                              // Number of Coefficients
    matrix[N, K] X;                             // Design Matrix

    // Design Matrix for the Coefficients
    X = P[, 2:D];
}

parameters {
    real alpha;                                 // Intercept
    vector[K] beta;                             // Coefficients
}

model {
    // Bernoulli Likelihood
    profile("Likelihood") {
        target += bernoulli_logit_glm_lpmf(Y | X, alpha, beta);
    }

    // Priors
    profile("Priors") {
        target += normal_lpdf(alpha | mu_alpha, sigma_alpha);
        target += normal_lpdf(beta | mu_beta, sigma_beta);
    }
}

generated quantities {
    array[N] int yrep;                          // Posterior Predictive Draws
    yrep = bernoulli_logit_rng(alpha + X * beta);
    
    vector[N] loglik;                           // Pointwise Log Likelihood
    for (n in 1:N) {
        loglik[n] = bernoulli_logit_lpmf(Y[n] | alpha + X[n] * beta);
    }
}

