#!/bin/bash
#SBATCH --job-name=amber_bench_cuda            # Job name
#SBATCH --mail-type=END,FAIL                   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=a.user@york.ac.uk          # Where to send mail 
#SBATCH --ntasks=40                            # Run a single task  
#SBATCH --cpus-per-task=1                      # Number of CPU cores per task
#SBATCH --mem=128gb                            # Job memory request
#SBATCH --time=12:00:00                        # Time limit hrs:min:sec
#SBATCH --output=logs/amber_bench_cuda_%j.log  # Standard output and error log
#SBATCH --partition=gpu                        # select the gpu nodes
#SBATCH --gres=gpu:1                           # select 1 gpu
 
echo "Running gaussian-test on $SLURM_CPUS_ON_NODE CPU cores"

python Dropout_6_Bernoulli_NoDropout.py