#!/usr/bin/python3
#SBATCH --job-name=edwin_test
#SBATCH --gres=gpu:1 -p compsci-gpu

python3 run_sonnet_gen.py
