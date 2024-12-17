#!/bin/bash
#
#SBATCH --partition=gpu_min8gb    # partition
#SBATCH --qos=gpu_min8gb          # QoS level
#SBATCH --job-name=heuristics_job # Job name
#SBATCH -o slurm_%x.%j.out        # File containing STDOUT output
#SBATCH -e slurm_%x.%j.err        # File containing STDERR output

python3 get_brkga_best_solution.py
