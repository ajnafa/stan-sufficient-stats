/* 
    Model: Categorical Likelihood with a Logit Link
    Author: A. Jordan Nafa
    Date: 2023-11-05
    License: MIT
*/

data {
    // Data Dimensions
    int<lower=1> N;                     // Number of Observations
    int<lower=2> K;                     // Number of Categories
    int<lower=1> D;                     // Number of Predictors

    // Input Data
    array[N] int<lower=0,upper=K> Y;    // Response Variable
    matrix[N, D] P;                     // Predictor Matrix
}

transformed data {
    int L = K - 1;                      // Number of Categories - 1
    int T = D - 1;                      // Number of Coefficients Per Category
    matrix[N, T] X;                     // Design Matrix

    // Design Matrix for the Coefficients
    X = P[, 2:D];

    // Zero Vector for the Reference Category
    vector[T] zeros = zeros_vector(T);
}

parameters {
    // Model Parameters
    vector[L] phi;                      // Intercept
    matrix[T, L] kappa;                 // Coefficients
}

transformed parameters {
    vector[K] alpha;                    // Intercept
    alpha = append_row(0, phi);

    matrix[T, K] beta;                  // Coefficients
    beta = append_col(zeros, kappa);
}

model {
    // Categorical Likelihood, Logit GLM
    profile("Likelihood") {
        target += categorical_logit_glm_lpmf(Y | X, alpha, beta);
    }

    // Priors
    profile("Priors") {
        target += std_normal_lpdf(phi);
        target += std_normal_lpdf(to_vector(kappa));
    }
}


