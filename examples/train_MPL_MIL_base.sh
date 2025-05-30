#!/bin/bash
#SBATCH --job-name=train_CLIP_MIL         # 作业名称
#SBATCH --output=log/output_%x_%j.log      # 输出文件名，%j代表作业ID
#SBATCH --error=log/error_%x_%j.log        # 错误文件名
#SBATCH -p RTX3090  ## 指定分区
#SBATCH -w gpu05  ## 节点id
#SBATCH -N 1        ## 使用节点数
#SBATCH -n 1        ## 任务数
#SBATCH --gres=gpu:1 ##申请gpu数量
#SBATCH --cpus-per-task=4

echo "SLURM_JOB_PARTITION={$SLURM_JOB_PARTITION}"
echo "SLURM_JOB_NODELIST={$SLURM_JOB_NODELIST}"
source activate wqh

export PYTHONPATH=./:$PYTHONPATH

cd ../

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_base/clip_meanpooling.yaml" \
                              --log_name "clip_meanpooling" --fold 5 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2" \
                              --feat_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"
