#!/bin/bash
#SBATCH --job-name=train_CLIP_MIL         # 作业名称
#SBATCH --output=log/output_%x_%j.log      # 输出文件名，%j代表作业ID
#SBATCH --error=log/error_%x_%j.log        # 错误文件名
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

cd ../

srun python train_TOP.py --config "/home/auwqh/code/CLIP-MIL/examples/config_TOP/TOP_s1.yaml" \
                              --log_name "TOP_s1" --fold 1 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/TOP" \
                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"

srun python train_TOP.py --config "/home/auwqh/code/CLIP-MIL/examples/config_TOP/TOP_s2.yaml" \
                              --log_name "TOP_s2" --fold 1 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/TOP" \
                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"

srun python train_TOP.py --config "/home/auwqh/code/CLIP-MIL/examples/config_TOP/TOP_s3.yaml" \
                              --log_name "TOP_s3" --fold 1 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/TOP" \
                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"

srun python train_TOP.py --config "/home/auwqh/code/CLIP-MIL/examples/config_TOP/TOP_s42.yaml" \
                              --log_name "TOP_s42" --fold 1 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/TOP" \
                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"


srun python train_TOP.py --config "/home/auwqh/code/CLIP-MIL/examples/config_TOP/TOP_s1.yaml" \
                              --log_name "TOP_s1" --fold 5 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2/TOP" \
                              --feat_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"

srun python train_TOP.py --config "/home/auwqh/code/CLIP-MIL/examples/config_TOP/TOP_s2.yaml" \
                              --log_name "TOP_s2" --fold 5 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2/TOP" \
                              --feat_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"

srun python train_TOP.py --config "/home/auwqh/code/CLIP-MIL/examples/config_TOP/TOP_s3.yaml" \
                              --log_name "TOP_s3" --fold 5 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2/TOP" \
                              --feat_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"

srun python train_TOP.py --config "/home/auwqh/code/CLIP-MIL/examples/config_TOP/TOP_s42.yaml" \
                              --log_name "TOP_s42" --fold 5 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2/TOP" \
                              --feat_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"