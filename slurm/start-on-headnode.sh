#!/usr/bin/env bash

head_node_ip=$(hostname --ip-address)
port=6379
ip_head="${head_node_ip}:${port}"
echo "IP head: ${ip_head}"
ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus 1 --block
