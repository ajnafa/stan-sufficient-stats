/* 
    Model: Hierarchical Bernoulli Likelihood with a Logit Link
    Author: A. Jordan Nafa
    Date: 2023-10-29
    License: MIT
*/

data {
    // Data Dimensions
    int<lower=0> N;                             // N observations
    int<lower=1> D;                             // K Predictors
    int<lower=1> J;                             // Number of Groups

    // Input Data
    array[N] int<lower=0,upper=1> Y;            // Binary Response 
    array[N] int<lower=1,upper=J> jj;           // Group Membership
    matrix[N,D] P;                              // Design Matrix

    // Priors for the Parameters
    real mu_alpha;                              // Intercept Prior Mean
    real<lower=0> sigma_alpha;                  // Intercept Prior Std Dev
    real mu_beta;                        // Coefficients Prior Mean
    real<lower=0> sigma_beta;                   // Coefficients Prior Std Dev
    real<lower=0> sigma_upsilon;                // Prior Std Dev for the Groups
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

    // Group Effects, Non-Centered Parameterization
    vector[J] upsilon_z;                        // Standardized Group Effects
    real<lower=0> tau;                          // Standard Deviation of Group Effects
}

transformed parameters {
    vector[J] upsilon;                          // Actual Group Effects
    upsilon = alpha + tau * upsilon_z;
}

model {
    // Bernoulli Likelihood
    profile("Likelihood") {
        target += bernoulli_logit_glm_lpmf(Y | X, upsilon[jj], beta);
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
    array[N] int yrep;                          // Posterior Predictive Draws
    yrep = bernoulli_logit_glm_rng(X, upsilon[jj], beta);
    
    vector[N] loglik;                           // Pointwise Log Likelihood
    for (n in 1:N) {
        loglik[n] = bernoulli_logit_lpmf(Y[n] | X[n] * beta + upsilon[jj[n]]);
    }
}