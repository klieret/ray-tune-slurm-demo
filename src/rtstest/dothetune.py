# Originally based on
# https://docs.ray.io/en/master/tune/getting-started.html#tune-tutorial

from __future__ import annotations

import copy
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hyperopt import hp
from ray import tune
from ray.air import CheckpointConfig, RunConfig
from ray.air.callbacks.wandb import WandbLoggerCallback

#  from ray.air.callbacks.mlflow import MLflowLoggerCallback
#  from ray.tune.logger import TBXLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ConvNet(nn.Module):
    def __init__(self, conf_out_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, conf_out_channels, kernel_size=3)
        self.fc = nn.Linear(64 * conf_out_channels, 10)
        self.conf_out_channels = conf_out_channels

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 64 * self.conf_out_channels)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    *,
    epoch_size=512,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # We set this just for the example to run quickly.
        if batch_idx * len(data) > epoch_size:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model: nn.Module, data_loader: DataLoader, *, test_size=256) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # We set this just for the example to run quickly.
            if batch_idx * len(data) > test_size:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


class Trainable(tune.Trainable):
    def setup(self, config):
        mnist_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        data_dir = Path("~/data").expanduser()
        self.train_loader = DataLoader(
            datasets.MNIST(
                str(data_dir), train=True, download=True, transform=mnist_transforms
            ),
            batch_size=64,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            datasets.MNIST(str(data_dir), train=False, transform=mnist_transforms),
            batch_size=64,
            shuffle=True,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ConvNet(conf_out_channels=config.get("conf_out_channels", 3)).to(
            device
        )

        self.optimizer = optim.SGD(
            self.model.parameters(), lr=config["lr"], momentum=config["momentum"]
        )

    def step(self):
        train(self.model, self.optimizer, self.train_loader)
        acc = test(self.model, self.test_loader)

        # Send the current training result back to Tune
        return dict(mean_accuracy=acc)

    def save_checkpoint(self, checkpoint_dir):
        path = Path(checkpoint_dir) / "checkpoint.pth"
        torch.save(self.model.state_dict(), path)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


@click.command()
@click.option(
    "--tune",
    "do_tune",
    help="Run Tune experiments",
    is_flag=True,
    show_default=True,
    default=False,
)
def main(do_tune=False):
    # Uncomment this to enable distributed execution
    # `ray.init(address="auto")`

    # Download the dataset first
    datasets.MNIST("~/data", train=True, download=True)

    run_config = RunConfig(
        callbacks=[
            # MLflowLoggerCallback(experiment_name="tune_experiment"),
            WandbLoggerCallback(
                api_key_file="~/.wandb_api_key", project="Wandb_example"
            ),
        ],
        sync_config=tune.SyncConfig(syncer=None),
        stop={"training_iteration": 20},
        checkpoint_config=CheckpointConfig(checkpoint_at_end=True),
    )

    train_config = dict(
        lr=1e-10,
        momentum=0.5,
        conf_out_channels=9,
    )

    if do_tune:
        space = copy.deepcopy(train_config)
        space.update(
            {
                "lr": hp.loguniform("lr", -10, -1),
                "momentum": hp.uniform("momentum", 0.1, 0.9),
                "conf_out_channels": hp.choice("conf_out_channels", [3, 6, 9]),
            }
        )
        hyperopt_search = HyperOptSearch(
            space, metric="mean_accuracy", mode="max", n_initial_points=40
        )

        tuner = tune.Tuner(
            Trainable,
            tune_config=tune.TuneConfig(
                scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
                num_samples=100,
                search_alg=hyperopt_search,
            ),
            run_config=run_config,
        )
        tuner.fit()
    else:
        tuner = tune.Tuner(
            Trainable,
            param_space=train_config,
            run_config=run_config,
        )
        tuner.fit()


if __name__ == "__main__":
    main()
