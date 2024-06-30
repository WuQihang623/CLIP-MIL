#!/bin/bash
#SBATCH --job-name=test_clip         # 作业名称
#SBATCH --output=output_%x_%j.log      # 输出文件名，%j代表作业ID
#SBATCH --error=error_%x_%j.log        # 错误文件名
#SBATCH -p RTX3090  ## 指定分区
#SBATCH -w gpu04  ## 节点id
#SBATCH -N 1        ## 使用节点数
#SBATCH -n 1        ## 任务数
#SBATCH --gres=gpu:1 ##申请gpu数量
#SBATCH --cpus-per-task=4

echo "SLURM_JOB_PARTITION={$SLURM_JOB_PARTITION}"
echo "SLURM_JOB_NODELIST={$SLURM_JOB_NODELIST}"
source activate wqh

export PYTHONPATH=./:$PYTHONPATH

wz=1
cd ../

CUDA_VISIBLE_DEVICES=0 srun python src/test_clip.py --data "/home/auwqh/code/CLIP-MIL/data/stained_dataset.json" \
--template_path "/home/auwqh/code/CLIP-MIL/src/templates/is_stained.json" \
--save_dir "/home/auwqh/code/CLIP-MIL/src/performance" \
--save_name "stain_classification"