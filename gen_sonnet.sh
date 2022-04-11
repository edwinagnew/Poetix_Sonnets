#!/bin/bash
#SBATCH --job-name=new_templates
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH -p compsci-gpu

source activate /home/home3/ea132/anaconda3/envs/poetix
python3 gen_poem_script.py
