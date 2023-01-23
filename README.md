<div align="center">
<h1>Ray tune & friends on SLURM</h1>
<p><em>Hyperparameter optimization tryout</em></p>
<p><a href="https://results.pre-commit.ci/latest/github/klieret/ray-tune-slurm-demo/main"><img src="https://results.pre-commit.ci/badge/github/klieret/ray-tune-slurm-demo/main.svg" alt="pre-commit.ci status"></a>
<a href="https://github.com/klieret/ray-tune-slurm-demo/actions"><img src="https://github.com/klieret/ray-tune-slurm-demo/actions/workflows/test.yml/badge.svg" alt="link checker"></a>
<a href="https://github.com/klieret/ray-tune-slurm-demo/blob/master/LICENSE.txt"><img src="https://img.shields.io/github/license/klieret/ray-tune-slurm-demo" alt="License"></a>
<a href="https://github.com/python/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black"></a></p>
<img src="readme_assets/scrot.png" width="500px"/>
</div>

## 📝 Description

This repository demonstrates/tests hyperparameter optimization with the following frameworks:

* [Ray tune][tune] as parent framework and to start jobs with [SLURM][slurm]
* [Optuna][optuna] to suggest the hyperaprameters
* [Wandb (weights & measures)][wandb] to log and visualize the results

> **Note**
> If you want to see this tech stack in an actual use case, see the [GNN tracking Hyperparameter Optimization repository][gnn-tracking-hpo].

## 📦 Installation

Use the conda environment, THEN `pip` install the package.

## 🔥 Running it!

### First test without batch system

* First run `src/rtstest/dothetune.py` (no batch submission) to also download the data file
  (because no internet connection on the compute nodes)

### Option 1: All-in-one

For a single batch jobs that uses multiple nodes to start both the head node and the works, see
`slurm/all-in-one`. While this is the example used in the ray documentation, it might not be
the best for most use cases, as it relies on having enough available nodes directly available
for enough time to complete all requested trials.

#### Live syncing to wandb

Because the compute nodes usually do not have internet, we need a separate tool for this.
See the documentation of [wandb-osh] for how to start the syncer on the head node.

### Option 2: Head node and worker nodes

Here, we start the ray head on the head (login) node and then use batch submission to start
worker nodes asynchronously.
Follow the following steps

1. Run `slurm/head_workers/start-on-headnode.sh` and note down the IP and redis password that are printed out
2. Submit several batch jobs `sbatch slurm/head_workers/start-on-worker.slurm <IP> <REDIS PWD>`
3. Start your tuning script on the head node: `slurm/head_workers/start-program.sh <IP> <REDIS PWD>`

> **Note**
> In my HPO scripts at [my main ML project][gnn-tracking-hpo] I instead write out the IP
> and password to files in my home directory and have dependent scripts read from there
> rather than passing them around on the command line.

Once the batch jobs for the workers start running, you should see activity in the tuning script output.

[tune]: https://docs.ray.io/en/master/tune/index.html
[tigergpu]: https://researchcomputing.princeton.edu/systems/tiger
[optuna]: https://optuna.org/
[wandb]: https://wandb.ai/site
[slurm]: https://slurm.schedmd.com/
[wandb-osh]: https://github.com/klieret/wandb-offline-sync-hook/
[gnn-tracking-hpo]: https://github.com/gnn-tracking/hyperparameter_optimization
