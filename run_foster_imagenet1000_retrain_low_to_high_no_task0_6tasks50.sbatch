#!/bin/bash
#SBATCH -p base_suma_rtx3090
#SBATCH --gres=gpu:2
#SBATCH -x node05,node04,node02
#SBATCH -t 72:00:00
#SBATCH -J foster_img1k_retrain_l2h_no0
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs

echo "JobID: $SLURM_JOB_ID"
echo "Host:  $(hostname)"
echo "Start: $(date)"

cd /home/yhha001/FOSTER

source ~/anaconda3/etc/profile.d/conda.sh
conda activate cil

echo "Python: $(which python)"
python -V
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"

python -u main.py --config ./configs/foster-imagenet1000-b0inc100-retrain-low-to-high-no-task0-6tasks50.json

echo "End:   $(date)"
