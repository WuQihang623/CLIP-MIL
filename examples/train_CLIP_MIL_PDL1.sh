#!/bin/bash
#SBATCH --job-name=train_CLIP_MIL         # 作业名称
#SBATCH --output=log/output_%x_%j.log      # 输出文件名，%j代表作业ID
#SBATCH --error=log/error_%x_%j.log        # 错误文件名
#SBATCH -p V100  ## 指定分区
#SBATCH -w gpu01  ## 节点id
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

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_PD_L1/clip_abmilpooling.yaml" \
                              --log_name "clip_abmilpooling" --fold 5 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/PDL1" \
                              --feat_dir "/home/auwqh/dataset/PDL1/meta_data/Testing/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/PDL1_fold"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_PD_L1/clip_meanpooling.yaml" \
                              --log_name "clip_meanpooling" --fold 5 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/PDL1" \
                              --feat_dir "/home/auwqh/dataset/PDL1/meta_data/Testing/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/PDL1_fold"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_PD_L1/clip_transmilpooling.yaml" \
                              --log_name "clip_transmilpooling" --fold 5 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/PDL1" \
                              --feat_dir "/home/auwqh/dataset/PDL1/meta_data/Testing/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/PDL1_fold"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_PD_L1/clip_instancepooling.yaml" \
                              --log_name "clip_instancepooling" --fold 5 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/PDL1" \
                              --feat_dir "/home/auwqh/dataset/PDL1/meta_data/Testing/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/PDL1_fold"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_PD_L1/clip_instancepooling_ensemble.yaml" \
                              --log_name "clip_instancepooling_ensemble" --fold 5 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/PDL1" \
                              --feat_dir "/home/auwqh/dataset/PDL1/meta_data/Testing/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/PDL1_fold"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_PD_L1/clip_stainpooling.yaml" \
                              --log_name "clip_stainpooling" --fold 5 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/PDL1" \
                              --feat_dir "/home/auwqh/dataset/PDL1/meta_data/Testing/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/PDL1_fold"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_PD_L1/clip_stainpooling_ensemble.yaml" \
                              --log_name "clip_stainpooling_ensemble" --fold 5 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/PDL1" \
                              --feat_dir "/home/auwqh/dataset/PDL1/meta_data/Testing/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/PDL1_fold"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_PD_L1/clip_instanceStainPooling_group.yaml" \
                              --log_name "clip_instanceStainPooling_group" --fold 5 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/PDL1" \
                              --feat_dir "/home/auwqh/dataset/PDL1/meta_data/Testing/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/PDL1_fold"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_PD_L1/clip_instanceStainPooling_group_ensemble.yaml" \
                              --log_name "clip_instanceStainPooling_group_ensemble" --fold 5 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/PDL1" \
                              --feat_dir "/home/auwqh/dataset/PDL1/meta_data/Testing/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/PDL1_fold"




