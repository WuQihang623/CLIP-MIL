#!/bin/bash
#SBATCH --job-name=extract_feature         # 作业名称
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

wz=1
cd ../

CUDA_VISIBLE_DEVICES=0 srun python extract_feature_fp.py --enc_name clip_ViTB32 \
--data_root /home/auwqh/dataset/HER2/patches_features_10x/ \
--data_slide_dir /home/auwqh/dataset/HER2/WSI/Testing/WSI/ \
--slide_ext ".tiff" \
--csv_path "/home/auwqh/dataset/HER2/patches_features_10x/process_list_autogen.csv"

CUDA_VISIBLE_DEVICES=0 srun python extract_feature_fp.py --enc_name clip_ViTB32 \
--data_root /home/auwqh/dataset/HER2/patches_features_5x/ \
--data_slide_dir /home/auwqh/dataset/HER2/WSI/Testing/WSI/ \
--slide_ext ".tiff" \
--csv_path "/home/auwqh/dataset/HER2/patches_features_5x/process_list_autogen.csv"

#CUDA_VISIBLE_DEVICES=0 srun python extract_feature_fp.py --enc_name clip_ViTB32 \
#--data_root /home/auwqh/dataset/WarwickHER2/patches_features_10x/ \
#--data_slide_dir /home/auwqh/dataset/WarwickHER2/WSI/IHC/ \
#--slide_ext ".ndpi" \
#--csv_path "/home/auwqh/dataset/WarwickHER2/patches_features_10x/process_list_autogen.csv"
#
#CUDA_VISIBLE_DEVICES=0 srun python extract_feature_fp.py --enc_name clip_ViTB32 \
#--data_root /home/auwqh/dataset/WarwickHER2/patches_features_5x/ \
#--data_slide_dir /home/auwqh/dataset/WarwickHER2/WSI/IHC/ \
#--slide_ext ".ndpi" \
#--csv_path "/home/auwqh/dataset/WarwickHER2/patches_features_5x/process_list_autogen.csv"