#!/usr/bin/env python3

# Might not actually be a bug but just needless constraints?
from __future__ import annotations

from functools import partial

from ray import tune
from ray.air import ScalingConfig
from ray.train.torch import TorchTrainer


def train_func_with_param(a=1, b=2):
    for i in range(3):
        tune.report(dict(i=i, a=a, b=b))


# Does not work
# ==================================================

trainer = TorchTrainer(
    train_func_with_param,
    scaling_config=ScalingConfig(num_workers=1),
)
trainer.fit()

# Does not work
# ==================================================

# Actually partial doesn't get rid of the parameters, it just sets/changes their default
trainer = TorchTrainer(
    partial(train_func_with_param, a=1, b=2),
    scaling_config=ScalingConfig(num_workers=1),
)
trainer.fit()

# Does work
# ==================================================


def train_func():
    return train_func_with_param(a=1, b=2)


trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(num_workers=1),
)
trainer.fit()
