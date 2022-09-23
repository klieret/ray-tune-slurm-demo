# Testing ray tune + hyperopt + mlflow/wandb on SLURM

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/klieret/ray-tune-slurm-test/main.svg)](https://results.pre-commit.ci/latest/github/klieret/ray-tune-slurm-test/main)
[![License](https://img.shields.io/github/license/klieret/ray-tune-slurm-test)](https://github.com/klieret/ray-tune-slurm-test/blob/master/LICENSE.txt)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

> **Warning**
> Experimental repository.

> **Warning**
> Testing has now mostly switched over to [the actual use case](github.com/klieret/gnn_tracking_experiments). See
> scripts there or PM me for more information.

## 📝 Description

Testing ray tune with hyperopt and reporting to mlflow/wandb with slurm batch submission on the the Princeton tigergpu cluster.

## 📦 Installation

Use the conda environment `pip` install the package.

## 🔥 Running it!

First run `src/rtstest/dothetest.py` (no batch submission) to also download the data file
(because no internet connection on the compute nodes), then use one of the
`*.sh` slurm files to run with batch system.
