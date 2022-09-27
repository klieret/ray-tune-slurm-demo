<div align="center">
<h1>Ray tune & friends on SLURM</h1>
<p><em>Hyperparameter optimization tryout</em></p>
<p><a href="https://results.pre-commit.ci/latest/github/klieret/ray-tune-slurm-test/main"><img src="https://results.pre-commit.ci/badge/github/klieret/ray-tune-slurm-test/main.svg" alt="pre-commit.ci status"></a>
<a href="https://github.com/klieret/ray-tune-slurm-test/actions"><img src="https://github.com/klieret/ray-tune-slurm-test/actions/workflows/test.yml/badge.svg" alt="link checker"></a>
<a href="https://github.com/klieret/ray-tune-slurm-test/blob/master/LICENSE.txt"><img src="https://img.shields.io/github/license/klieret/ray-tune-slurm-test" alt="License"></a>
<a href="https://github.com/python/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black"></a></p>
</div>

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
