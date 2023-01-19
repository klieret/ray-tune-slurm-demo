#!/bin/bash

#SBATCH --job-name=ray_worker.sh
#SBATCH --output=ray_worker-%j.log

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpu:1

ip_head=$1
redis_password=$2
echo "Using head node at ${ip_head}"
echo "Redis password ${redis_password}"
ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "1" --block --redis-password="${redis_password}"
