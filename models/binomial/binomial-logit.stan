/* 
    Model: Binomial Likelihood with a Logit Link, Sufficient
           Formulation of the Bernoulli Likelihood
    Author: A. Jordan Nafa
    Date: 2023-10-29
    License: MIT
*/

data {
    // Data Dimensions
    int<lower=0> N;                             // N Cells
    int<lower=1> D;                             // D Predictors

    // Input Data
    array[N] int<lower=0> Y;                    // Observed Successes
    array[N] int<lower=0> K;                    // Observed Trials
    matrix[N,D] P;                              // Design Matrix

    // Priors for the Parameters
    real mu_alpha;                              // Intercept Prior Mean
    real<lower=0> sigma_alpha;                  // Intercept Prior Std Dev
    real mu_beta;                               // Coefficients Prior Mean
    real<lower=0> sigma_beta;                   // Coefficients Prior Std Dev
}

transformed data {
    int L = D - 1;                              // Number of Coefficients
    matrix[N, L] X;                             // Design Matrix

    // Design Matrix for the Coefficients
    X = P[, 2:D];
}

parameters {
    real alpha;                                 // Intercept
    vector[L] beta;                             // Coefficients
}

transformed parameters {
    // Linear Predictor on the Logit Scale
    vector[N] mu;
    mu = alpha + X * beta;
}

model {
    // Binomial Likelihood
    profile("Likelihood") {
        target += binomial_logit_lpmf(Y | K, mu);
    }

    // Priors
    profile("Priors") {
        target += normal_lpdf(alpha | mu_alpha, sigma_alpha);
        target += normal_lpdf(beta | mu_beta, sigma_beta);
    }
}

generated quantities {
    vector[N] theta;                            // Probability of Success
    theta = inv_logit(mu);

    array[N] int yrep;                          // Posterior Predictive Draws
    yrep = binomial_rng(K, theta);
    
    vector[N] loglik;                           // Pointwise Log Likelihood
    for (n in 1:N) {
        loglik[n] = binomial_logit_lpmf(Y[n] | K[n], mu[n]);
    }
}