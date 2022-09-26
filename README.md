# Testing ray tune + hyperopt + mlflow/wandb on SLURM

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/klieret/ray-tune-slurm-test/main.svg)](https://results.pre-commit.ci/latest/github/klieret/ray-tune-slurm-test/main)
[![License](https://img.shields.io/github/license/klieret/ray-tune-slurm-test)](https://github.com/klieret/ray-tune-slurm-test/blob/master/LICENSE.txt)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

> **Warning**
> Experimental repository.

> **Warning**
> Development has now mostly switched over to [the actual use case](github.com/klieret/gnn_tracking_experiments). See
> scripts there or PM me for more information.

## 📝 Description

Running hyperparameter optimization with the following stack of tools:

* Ray tune as parent framework and to start jobs with SLURM (on the Princeton `tigergpu` cluster)
* Optuna to suggest the hyperaprameters
* Wandb to log and visualize the results

## 📦 Installation

Use the conda environment, then `pip` install the package.

## 🔥 Running it!

First run `src/rtstest/dothetest.py` (no batch submission) to also download the data file
(because no internet connection on the compute nodes), then use one of the
`*.sh` slurm files to run with batch system.
