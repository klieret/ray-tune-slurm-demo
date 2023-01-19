#!/usr/bin/env bash

ip_head=$1
export ip_head
redis_password=$2
export redis_password
echo "Redis password ${redis_password}"
echo "IP Head: $ip_head"

export WANDB_MODE=offline
echo $(which python3)
python3 ../../src/rtstest/dothetune.py --tune --gpu
