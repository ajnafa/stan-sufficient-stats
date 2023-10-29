
# Stan Models for Scalable Bayesian Inference Using Sufficient Statistics

[A. Jordan Nafa](https://www.ajordannafa.com/)

------------------------------------------------------------------------

This github repository contains Stan code for typical and sufficient
statistic-based implementations of Bayesian models for various
likelihoods. Sufficient statistics are a core characteristic of
likelihoods that belong to the [exponential
family](https://en.wikipedia.org/wiki/Exponential_family) and can be
exploited to drastically reduce the computational cost of Bayesian
inference. For a general overview of sufficient statistics in Stan, see
section 25.9 of the [Stan User’s
Guide](https://mc-stan.org/docs/stan-users-guide/exploiting-sufficient-statistics.html).

To download the development version of this repository, you can execute
the following command from a desktop terminal

    git clone -b development https://github.com/ajnafa/stan-sufficient-stats.git

This repository is a work in progress. If there is a specific likelihood
or model you would like an implementation for, please open an issue with
a description of the model and preferably a link to a paper or other
resource that describes the model.

## Licenses

All code in this repository is provided under an [MIT
License](LICENSE.md).
