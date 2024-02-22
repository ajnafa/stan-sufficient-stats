/* 
    Model: Hierarchical Bernoulli Likelihood with a Logit Link,
           Varying Intercepts Over Groups, Fixed Slopes
    Author: A. Jordan Nafa
    Date: 2024-02-21
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
    vector[D-1] mu_beta;                        // Coefficients Prior Mean
    vector<lower=0>[D-1] sigma_beta;            // Coefficients Prior Std Dev
    real<lower=0> sigma_upsilon;                // Group Effects Prior Std Dev

    // Additional Arguments
    int<lower=2, upper=D> treat_idx;            // Index of the Treatment Variable
    int<lower=1> G;                             // Number of New Groups to Simulate
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
    real AME;                            // Average Marginal Effect
    real EY1;                            // Expected Value of Y(1, Z)
    real EY0;                            // Expected Value of Y(0, Z)

    /* Calculate Quantities in curly brackets so we aren't writing the 
    entire 250K observation array to the CSV at every iteration */
    {
        // Simulate New Groups from the Implicit Multivariate Normal
        array[J] real u_sim = normal_rng(rep_vector(0, J), 1);
        vector[J] upsilon_sim = alpha + tau * to_vector(u_sim);

        // Simulate Counterfactual Outcomes
        array[N] real Y0 = bernoulli_logit_glm_rng(X0, upsilon_sim[jj], beta);
        array[N] real Y1 = bernoulli_logit_glm_rng(X1, upsilon_sim[jj], beta);

        // Quantities of Interest
        EY0 = mean(Y0);
        EY1 = mean(Y1);
        AME = EY1 - EY0;
    }
}