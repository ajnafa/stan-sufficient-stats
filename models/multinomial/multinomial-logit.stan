/* 
    Model: Multinomial Likelihood with a Logit Link
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
    array[N] int<lower=1,upper=D> dd;   // Assignment Mapping
}

transformed data {
    int L = K - 1;                      // Number of Categories - 1

    // Calculate Sufficent Statistics for the Multinomial Distribution
    array[K, D] int<lower=0> y = rep_array(0, K, D);
    for (n in 1:N) {
        y[Y[n], dd[n]] += 1;
    }

    // Zero Vector for the Reference Category
    vector[D] zeros = zeros_vector(D);
}

parameters {
    matrix[D, L] kappa;                 // Coefficients
}

transformed parameters {
    /* Coefficients for the Reference Category of the Response 
    are fixed to zero for identifiability of the likelihood */
    matrix[D, K] beta = append_col(zeros, kappa);
}

model {
    // Multinomial Likelihood
    profile("Likelihood") {
        for (d in 1:D) {
            target += multinomial_logit_lpmf(y[, d] | beta[d]');
        }
    }

    // Priors
    profile("Priors") {
        target += std_normal_lpdf(to_vector(kappa));
    }
}
