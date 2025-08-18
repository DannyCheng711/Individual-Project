#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=gpgpuC
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cc2224

export PATH=/vol/bitbucket/${USER}/venv/bin:$PATH
source activate
. /vol/cuda/12.0.0/setup.sh

/usr/bin/nvidia-smi
uptime

cd /vol/bitbucket/${USER}/IndividualProject/Poc
python -u -m vocmain.main 2>&1 | tee -a "logs/train_$(date +%F_%H-%M)_${SLURM_JOB_ID}.log"
