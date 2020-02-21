#!/usr/bin/python3
#SBATCH --job-name=edwin_test
#SBATCH -t 00:05:00
#SBATCH --gres=gpu:1
#SBATCH -p compsci-gpu

import sonnet_basic
import argparse

s = sonnet_basic.Sonnet_Gen()
parser = argparse.ArgumentParser()
parser.add_argument('--prompt', '-p', required=True)

#args = parser.parse_args()

poem = s.gen_poem_edwin("death", print_poem=True)
