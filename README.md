# Hyperparameter optimization with ray tune & friends on SLURM

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/klieret/ray-tune-slurm-test/main.svg)](https://results.pre-commit.ci/latest/github/klieret/ray-tune-slurm-test/main)
[![License](https://img.shields.io/github/license/klieret/ray-tune-slurm-test)](https://github.com/klieret/ray-tune-slurm-test/blob/master/LICENSE.txt)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

## 📝 Description

This repository demonstrates/tests hyperparameter optimization with the following frameworks:

* [Ray tune][tune] as parent framework and to start jobs with [SLURM][slurm] (on the [Princeton `tigergpu` cluster][tigergpu])
* [Optuna][optuna] to suggest the hyperaprameters
* [Wandb (weights & measures)][wandb] to log and visualize the results

> **Note**
> If you want to see this in an actual use case, see the [GNN tracking repository](https://github.com/klieret/gnn_tracking_experiments).

## 📦 Installation

Use the conda environment, then `pip` install the package.

## 🔥 Running it!

First run `src/rtstest/dothetest.py` (no batch submission) to also download the data file
(because no internet connection on the compute nodes), then use one of the
`*.sh` slurm files to run with batch system.

[tune]: https://docs.ray.io/en/master/tune/index.html
[tigergpu]: https://researchcomputing.princeton.edu/systems/tiger
[optuna]: https://optuna.org/
[wandb]: https://wandb.ai/site
[slurm]: https://slurm.schedmd.com/
