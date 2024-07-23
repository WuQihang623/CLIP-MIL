#!/bin/bash
#SBATCH --job-name=train_MIL         # 作业名称
#SBATCH --output=log/output_%x_%j.log      # 输出文件名，%j代表作业ID
#SBATCH --error=log/error_%x_%j.log        # 错误文件名
#SBATCH -p A100  ## 指定分区
#SBATCH -w gpu09  ## 节点id
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


#### HER2

srun python train_MIL.py --model_name "ABMIL" \
--feature_dim 512 --feat_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/" \
--fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold/" --batch_size 1 \
--n_epochs 50 --workers 4 --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/HER2" \
--log_name "ABMIL_clip_ViTB32_weights" --lr 0.0001 --step_size 15 --gamma 0.1 --n_classes 4



srun python train_MIL.py --model_name "CLAM_MB" \
--feature_dim 512 --feat_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/" \
--fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold/" --batch_size 1 \
--n_epochs 50 --workers 4 --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/HER2" \
--log_name "CLAM_MB_clip_ViTB32_weights" --lr 0.0001 --step_size 15 --gamma 0.1 --n_classes 4



srun python train_MIL.py --model_name "TransMIL" \
--feature_dim 512 --feat_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/" \
--fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold/" --batch_size 1 \
--n_epochs 50 --workers 4 --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/HER2" \
--log_name "TransMIL_clip_ViTB32_weights" --lr 0.0001 --step_size 15 --gamma 0.1 --n_classes 4




#### PDL1
#srun python train_MIL.py --model_name "ABMIL" \
#--feature_dim 512 --feat_dir "/home/auwqh/dataset/PDL1/meta_data/Testing/patch/clip_ViTB32/pt_files/" \
#--fold_dir "/home/auwqh/code/CLIP-MIL/data/PDL1_fold" --batch_size 1 \
#--n_epochs 50 --workers 4 --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/PDL1" \
#--log_name "ABMIL_clip_ViTB32_weights" --lr 0.0001 --step_size 15 --gamma 0.1 --n_classes 3
#
#srun python train_MIL.py --model_name "ABMIL" \
#--feature_dim 1024 --feat_dir "/home/auwqh/dataset/PDL1/meta_data/Testing/patch/resnet50_trunc/pt_files/" \
#--fold_dir "/home/auwqh/code/CLIP-MIL/data/PDL1_fold/" --batch_size 1 \
#--n_epochs 50 --workers 4 --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/PDL1" \
#--log_name "ABMIL_resnet50_imagenet_weights" --lr 0.0001 --step_size 15 --gamma 0.1 --n_classes 3
#
#
#
#srun python train_MIL.py --model_name "CLAM_MB" \
#--feature_dim 512 --feat_dir "/home/auwqh/dataset/PDL1/meta_data/Testing/patch/clip_ViTB32/pt_files/" \
#--fold_dir "/home/auwqh/code/CLIP-MIL/data/PDL1_fold/" --batch_size 1 \
#--n_epochs 50 --workers 4 --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/PDL1" \
#--log_name "CLAM_MB_clip_ViTB32_weights" --lr 0.0001 --step_size 15 --gamma 0.1 --n_classes 3
#
#srun python train_MIL.py --model_name "CLAM_MB" \
#--feature_dim 1024 --feat_dir "/home/auwqh/dataset/PDL1/meta_data/Testing/patch/resnet50_trunc/pt_files/" \
#--fold_dir "/home/auwqh/code/CLIP-MIL/data/PDL1_fold/" --batch_size 1 \
#--n_epochs 50 --workers 4 --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/PDL1" \
#--log_name "CLAM_MB_resnet50_imagenet_weights" --lr 0.0001 --step_size 15 --gamma 0.1 --n_classes 3
#
#
#
#srun python train_MIL.py --model_name "TransMIL" \
#--feature_dim 512 --feat_dir "/home/auwqh/dataset/PDL1/meta_data/Testing/patch/clip_ViTB32/pt_files/" \
#--fold_dir "/home/auwqh/code/CLIP-MIL/data/PDL1_fold/" --batch_size 1 \
#--n_epochs 50 --workers 4 --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/PDL1" \
#--log_name "TransMIL_clip_ViTB32_weights" --lr 0.0001 --step_size 15 --gamma 0.1 --n_classes 3
#
#srun python train_MIL.py --model_name "TransMIL" \
#--feature_dim 1024 --feat_dir "/home/auwqh/dataset/PDL1/meta_data/Testing/patch/resnet50_trunc/pt_files/" \
#--fold_dir "/home/auwqh/code/CLIP-MIL/data/PDL1_fold/" --batch_size 1 \
#--n_epochs 50 --workers 4 --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/PDL1" \
#--log_name "TransMIL_resnet50_imagenet_weights" --lr 0.0001 --step_size 15 --gamma 0.1 --n_classes 3


### Warwick

srun python train_MIL.py --model_name "ABMIL" \
--feature_dim 512 --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
--fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick/" --batch_size 1 --fold 1 \
--n_epochs 50 --workers 4 --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2" \
--log_name "ABMIL_clip_ViTB32_weights" --lr 0.0001 --step_size 15 --gamma 0.1 --n_classes 4


srun python train_MIL.py --model_name "CLAM_MB" \
--feature_dim 512 --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
--fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick/" --batch_size 1 --fold 1 \
--n_epochs 50 --workers 4 --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2" \
--log_name "CLAM_MB_clip_ViTB32_weights" --lr 0.0001 --step_size 15 --gamma 0.1 --n_classes 4


srun python train_MIL.py --model_name "TransMIL" \
--feature_dim 512 --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
--fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick/" --batch_size 1 --fold 1 \
--n_epochs 50 --workers 4 --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2" \
--log_name "TransMIL_clip_ViTB32_weights" --lr 0.0001 --step_size 15 --gamma 0.1 --n_classes 4
