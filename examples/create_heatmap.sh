#!/bin/bash
#SBATCH --job-name=create_heatmap         # 作业名称
#SBATCH --output=./log/output_%x_%j.log      # 输出文件名，%j代表作业ID
#SBATCH --error=./log/error_%x_%j.log        # 错误文件名
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

srun python create_heatmap.py --config "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_instance_ensemble/config.yaml" \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_instance_ensemble" \
--description "instance" --device "cuda" \
--csv_dir "/home/auwqh/code/CLIP-MIL/data/Warwick" \
--fold 1 --wsi_dir "/home/auwqh/dataset/WarwickHER2/WSI/IHC/" \
--h5_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/h5_files/" \
--save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_instance_ensemble/heatmap"

srun python create_heatmap.py --config "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_stain_ensemble/config.yaml" \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_stain_ensemble/" \
--description "stain" --device "cuda" \
--csv_dir "/home/auwqh/code/CLIP-MIL/data/Warwick" \
--fold 1 --wsi_dir "/home/auwqh/dataset/WarwickHER2/WSI/IHC/" \
--h5_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/h5_files/" \
--save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_stain_ensemble/heatmap"

srun python create_heatmap.py --config "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_concat_ensemble/config.yaml" \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_concat_ensemble/" \
--description "instance" --device "cuda" \
--csv_dir "/home/auwqh/code/CLIP-MIL/data/Warwick" \
--fold 1 --wsi_dir "/home/auwqh/dataset/WarwickHER2/WSI/IHC/" \
--h5_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/h5_files/" \
--save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_concat_ensemble/heatmap_cls"

srun python create_heatmap.py --config "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_concat_ensemble/config.yaml" \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_concat_ensemble/" \
--description "stain" --device "cuda" \
--csv_dir "/home/auwqh/code/CLIP-MIL/data/Warwick" \
--fold 1 --wsi_dir "/home/auwqh/dataset/WarwickHER2/WSI/IHC/" \
--h5_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/h5_files/" \
--save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_concat_ensemble/heatmap_stn"

srun python create_heatmap.py --config "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_group_ensemble/config.yaml" \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_group_ensemble/" \
--description "instance" --device "cuda" \
--csv_dir "/home/auwqh/code/CLIP-MIL/data/Warwick" \
--fold 1 --wsi_dir "/home/auwqh/dataset/WarwickHER2/WSI/IHC/" \
--h5_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/h5_files/" \
--save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_group_ensemble/heatmap_cls"

srun python create_heatmap.py --config "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_group_ensemble/config.yaml" \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_group_ensemble/" \
--description "stain" --device "cuda" \
--csv_dir "/home/auwqh/code/CLIP-MIL/data/Warwick" \
--fold 1 --wsi_dir "/home/auwqh/dataset/WarwickHER2/WSI/IHC/" \
--h5_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/h5_files/" \
--save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_group_ensemble/heatmap_stn"

srun python create_heatmap.py --config "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_group_ensemble1/config.yaml" \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_group_ensemble1/" \
--description "instance" --device "cuda" \
--csv_dir "/home/auwqh/code/CLIP-MIL/data/Warwick" \
--fold 1 --wsi_dir "/home/auwqh/dataset/WarwickHER2/WSI/IHC/" \
--h5_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/h5_files/" \
--save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_group_ensemble1/heatmap_cls"

srun python create_heatmap.py --config "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_group_ensemble1/config.yaml" \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_group_ensemble1/" \
--description "stain" --device "cuda" \
--csv_dir "/home/auwqh/code/CLIP-MIL/data/Warwick" \
--fold 1 --wsi_dir "/home/auwqh/dataset/WarwickHER2/WSI/IHC/" \
--h5_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/h5_files/" \
--save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_group_ensemble1/heatmap_stn"

srun python create_heatmap.py --config "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_instance_ensemble/config.yaml" \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_instance_ensemble/" \
--description "instance" --device "cuda" \
--csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
--fold 5 --wsi_dir "/home/auwqh/dataset/HER2/WSI/Testing/WSI/" \
--h5_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/h5_files/" \
--save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_instance_ensemble/heatmap" \
--ext ".tiff"

srun python create_heatmap.py --config "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_stain_ensemble/config.yaml" \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_stain_ensemble/" \
--description "stain" --device "cuda" \
--csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
--fold 5 --wsi_dir "/home/auwqh/dataset/HER2/WSI/Testing/WSI/" \
--h5_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/h5_files/" \
--save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_stain_ensemble/heatmap" \
--ext ".tiff"

srun python create_heatmap.py --config "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_concat_ensemble/config.yaml" \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_concat_ensemble" \
--description "instance" --device "cuda" \
--csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
--fold 5 --wsi_dir "/home/auwqh/dataset/HER2/WSI/Testing/WSI/" \
--h5_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/h5_files/" \
--save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_concat_ensemble/heatmap_cls" \
--ext ".tiff"

srun python create_heatmap.py --config "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_concat_ensemble/config.yaml" \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_concat_ensemble" \
--description "stain" --device "cuda" \
--csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
--fold 5 --wsi_dir "/home/auwqh/dataset/HER2/WSI/Testing/WSI/" \
--h5_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/h5_files/" \
--save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_concat_ensemble/heatmap_stn" \
--ext ".tiff"

srun python create_heatmap.py --config "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_group_ensemble/config.yaml" \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_group_ensemble" \
--description "instance" --device "cuda" \
--csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
--fold 5 --wsi_dir "/home/auwqh/dataset/HER2/WSI/Testing/WSI/" \
--h5_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/h5_files/" \
--save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_group_ensemble/heatmap_cls" \
--ext ".tiff"

srun python create_heatmap.py --config "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_group_ensemble/config.yaml" \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_group_ensemble" \
--description "stain" --device "cuda" \
--csv_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold" \
--fold 5 --wsi_dir "/home/auwqh/dataset/HER2/WSI/Testing/WSI/" \
--h5_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/h5_files/" \
--save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_group_ensemble/heatmap_stn" \
--ext ".tiff"