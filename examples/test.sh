#!/bin/bash
#SBATCH --job-name=test_MIL         # 作业名称
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

cd ../

#### train ZJY ————> test ZJY
#srun python test.py --model "ABMIL" --num_classes 4 --feat_dim 512 \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/HER2/ABMIL_clip_ViTB32_weights" \
#--train_fold 5 --test_fold -1 --save_name "ZJY2ZJY" --csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
#--pt_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/"
#
#srun python test.py --model "CLAM" --num_classes 4 --feat_dim 512 \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/HER2/CLAM_MB_clip_ViTB32_weights" \
#--train_fold 5 --test_fold -1 --save_name "ZJY2ZJY" --csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
#--pt_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/"
#
#srun python test.py --model "TransMIL" --num_classes 4 --feat_dim 512 \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/HER2/TransMIL_clip_ViTB32_weights" \
#--train_fold 5 --test_fold -1 --save_name "ZJY2ZJY" --csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
#--pt_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/"
#
#srun python test.py --model "CLIPMIL" \
#--config "/home/auwqh/code/CLIP-MIL/examples/config_HER2/clip_abmilpooling.yaml" \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/HER2/clip_abmilpooling" \
#--train_fold 5 --test_fold -1 --save_name "ZJY2ZJY" --csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
#--pt_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/"
#
#srun python test.py --model "CLIPMIL" \
#--config "/home/auwqh/code/CLIP-MIL/examples/config_HER2/clip_meanpooling.yaml" \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/HER2/clip_meanpooling" \
#--train_fold 5 --test_fold -1 --save_name "ZJY2ZJY" --csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
#--pt_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/"
#
#srun python test.py --model "CLIPMIL" \
#--config "/home/auwqh/code/CLIP-MIL/examples/config_HER2/clip_transmilpooling.yaml" \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/HER2/clip_transmilpooling" \
#--train_fold 5 --test_fold -1 --save_name "ZJY2ZJY" --csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
#--pt_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/"
#
#srun python test.py --model "CLIPMIL" \
#--config "/home/auwqh/code/CLIP-MIL/examples/config_HER2/clip_instancepooling.yaml" \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/HER2/clip_instancepooling" \
#--train_fold 5 --test_fold -1 --save_name "ZJY2ZJY" --csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
#--pt_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/"
#
#srun python test.py --model "CLIPMIL" \
#--config "/home/auwqh/code/CLIP-MIL/examples/config_HER2/clip_instancepooling_ensemble.yaml" \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/HER2/clip_instancepooling" \
#--train_fold 5 --test_fold -1 --save_name "ZJY2ZJY_Ensemble" --csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
#--pt_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/"
#
#srun python test.py --model "CLIPMIL" \
#--config "/home/auwqh/code/CLIP-MIL/examples/config_HER2/clip_stainpooling.yaml" \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/HER2/clip_stainpooling" \
#--train_fold 5 --test_fold -1 --save_name "ZJY2ZJY" --csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
#--pt_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/"
#
#srun python test.py --model "CLIPMIL" \
#--config "/home/auwqh/code/CLIP-MIL/examples/config_HER2/clip_stainpooling_ensemble.yaml" \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/HER2/clip_stainpooling" \
#--train_fold 5 --test_fold -1 --save_name "ZJY2ZJY_Ensemble" --csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
#--pt_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/"
#
#srun python test.py --model "CLIPMIL" \
#--config "/home/auwqh/code/CLIP-MIL/examples/config_HER2/clip_instanceStainPooling.yaml" \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/HER2/clip_instanceStainPooling" \
#--train_fold 5 --test_fold -1 --save_name "ZJY2ZJY" --csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
#--pt_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/"
#
#srun python test.py --model "CLIPMIL" \
#--config "/home/auwqh/code/CLIP-MIL/examples/config_HER2/clip_instanceStainPooling_ensemble.yaml" \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/HER2/clip_instanceStainPooling" \
#--train_fold 5 --test_fold -1 --save_name "ZJY2ZJY_Ensemble" --csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
#--pt_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/"
#
#srun python test.py --model "CLIPMIL" \
#--config "/home/auwqh/code/CLIP-MIL/examples/config_HER2/clip_instanceStainPooling_group.yaml" \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/HER2/clip_instanceStainPooling_group" \
#--train_fold 5 --test_fold -1 --save_name "ZJY2ZJY" --csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
#--pt_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/"
#
#srun python test.py --model "CLIPMIL" \
#--config "/home/auwqh/code/CLIP-MIL/examples/config_HER2/clip_instanceStainPooling_group_ensemble.yaml" \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/HER2/clip_instanceStainPooling_group" \
#--train_fold 5 --test_fold -1 --save_name "ZJY2ZJY_Ensemble" --csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
#--pt_dir "/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/pt_files/"

#### Train ZJY ----> test Warwick
#srun python test.py --model "ABMIL" --num_classes 4 --feat_dim 512 \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/ABMIL_clip_ViTB32_weights" \
#--train_fold 5 --test_fold 1 --save_name "ZJY2Warwick" --csv_dir "/home/auwqh/code/CLIP-MIL/data/Warwick" \
#--pt_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/"
#
#srun python test.py --model "CLAM" --num_classes 4 --feat_dim 512 \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/CLAM_MB_clip_ViTB32_weights" \
#--train_fold 5 --test_fold 1 --save_name "ZJY2Warwick" --csv_dir "/home/auwqh/code/CLIP-MIL/data/Warwick" \
#--pt_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/"
#
#srun python test.py --model "TransMIL" --num_classes 4 --feat_dim 512 \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/TransMIL_clip_ViTB32_weights" \
#--train_fold 5 --test_fold 1 --save_name "ZJY2Warwick" --csv_dir "/home/auwqh/code/CLIP-MIL/data/Warwick" \
#--pt_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/"
#
#srun python test.py --model "CLIPMIL" \
#--config "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_group_ensemble/config.yaml" \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_group_ensemble" \
#--train_fold 5 --test_fold 1 --save_name "ZJY2Warwick" --csv_dir "/home/auwqh/code/CLIP-MIL/data/Warwick" \
#--pt_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/"


### Train Warwic ----> test Warwick


### Train Warwic ----> Test ZJY
srun python test.py --model "ABMIL" --num_classes 4 --feat_dim 512 \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/ABMIL_clip_ViTB32_weights" \
--train_fold 1 --test_fold 5 --save_name "Warwick2ZJY" --csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
--pt_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/"

srun python test.py --model "CLAM" --num_classes 4 --feat_dim 512 \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/CLAM_MB_clip_ViTB32_weights" \
--train_fold 1 --test_fold 5 --save_name "Warwick2ZJY" --csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
--pt_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/"

srun python test.py --model "TransMIL" --num_classes 4 --feat_dim 512 \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/TransMIL_clip_ViTB32_weights" \
--train_fold 1 --test_fold 5 --save_name "Warwick2ZJY" --csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
--pt_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/"

srun python test.py --model "CLIPMIL" \
--config "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_group_ensemble/config.yaml" \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_group_ensemble" \
--train_fold 1 --test_fold 5 --save_name "Warwick2ZJY" --csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
--pt_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/"
