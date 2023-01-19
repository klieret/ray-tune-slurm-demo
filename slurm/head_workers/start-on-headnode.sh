#!/usr/bin/env bash

head_node_ip=$(hostname --ip-address)
port=6379
ip_head="${head_node_ip}:${port}"
redis_password=$(uuidgen)
echo "Redis password ${redis_password}"
export redis_password

export ip_head
echo "IP head: ${ip_head}"
ray start -vvv --head --node-ip-address="$head_node_ip" --port=$port --num-cpus 1 --num-gpus 0 --block --dashboard-host=0.0.0.0 --dashboard-port=8841
