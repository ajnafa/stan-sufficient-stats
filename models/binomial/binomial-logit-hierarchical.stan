/* 
    Model: Hierarchical Binomial Likelihood with a Logit Link,
           Sufficient Formulation, Non-Centered Parameterization
    Author: A. Jordan Nafa
    Date: 2023-10-29
    License: MIT
*/

data {
    // Data Dimensions
    int<lower=0> N;                             // N Cells
    int<lower=1> D;                             // D Predictors
    int<lower=1> J;                             // Number of Groups

    // Input Data
    array[N] int<lower=0> Y;                    // Observed Successes
    array[N] int<lower=0> K;                    // Observed Trials
    array[N] int<lower=1,upper=J> jj;           // Group Membership
    matrix[N,D] P;                              // Design Matrix

    // Priors for the Parameters
    real mu_alpha;                              // Intercept Prior Mean
    real<lower=0> sigma_alpha;                  // Intercept Prior Std Dev
    real mu_beta;                               // Coefficients Prior Mean
    real<lower=0> sigma_beta;                   // Coefficients Prior Std Dev
    real<lower=0> sigma_upsilon;                // Prior Std Dev for the Groups
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

    // Group Effects, Non-Centered Parameterization
    vector[J] upsilon_z;                        // Standardized Group Effects
    real<lower=0> tau;                          // Standard Deviation of Group Effects
}

transformed parameters {
    vector[J] upsilon;                          // Actual Group Effects
    upsilon = alpha + tau * upsilon_z;

    // Linear Predictor on the Logit Scale
    vector[N] mu;
    mu = X * beta + upsilon[jj];
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
        target += std_normal_lpdf(upsilon_z);
        target += normal_lpdf(tau | 0, sigma_upsilon) 
            - 1 * normal_lccdf(0 | 0, sigma_upsilon);
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