#!/usr/bin/bash
#SBATCH --job-name=edwin_test
#SBATCH -t 00:05:00
#SBATCH --gres=gpu:1
#SBATCH -p compsci-gpu

python3 run_sonnet_gen.py
