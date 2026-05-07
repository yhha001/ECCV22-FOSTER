#!/bin/bash
#SBATCH -p base_suma_rtx3090
#SBATCH -x node05,node04,node02
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -J foster_img100_b0inc10
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

cd /home/yhha001/FOSTER || exit 1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate cil

echo "Python: $(which python)"
python -V
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"

python -u main.py --config ./configs/foster-imagenet100-b0inc10.json
