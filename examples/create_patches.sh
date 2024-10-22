#!/bin/bash
#SBATCH --job-name=create_patches         # 作业名称
#SBATCH --output=./log/output_%x_%j.log      # 输出文件名，%j代表作业ID
#SBATCH --error=./log/error_%x_%j.log        # 错误文件名
#SBATCH -p cpu  ## 指定分区
#SBATCH -w gpu06  ## 节点id
#SBATCH -N 1        ## 使用节点数
#SBATCH -n 1        ## 任务数
#SBATCH --gres=gpu:0 ##申请gpu数量
#SBATCH --cpus-per-task=4

echo "SLURM_JOB_PARTITION={$SLURM_JOB_PARTITION}"
echo "SLURM_JOB_NODELIST={$SLURM_JOB_NODELIST}"
source activate wqh

export PYTHONPATH=./:$PYTHONPATH

wz=1
cd ../

#srun python create_patches_fp.py --source "/home/auwqh/dataset/HER2/WSI/Testing/WSI/*.tiff" \
#                                 --step_size 256 --patch_size 256 --patch --seg \
#                                 --save_dir /home/auwqh/dataset/HER2/patches_features_10x/ \
#                                 --patch_level 2 \
#                                 --ann_positive /home/auwqh/dataset/HER2/control_group/
#
#srun python create_patches_fp.py --source "/home/auwqh/dataset/HER2/WSI/Testing/WSI/*.tiff" \
#                                 --step_size 256 --patch_size 256 --patch --seg \
#                                 --save_dir /home/auwqh/dataset/HER2/patches_features_5x/ \
#                                 --patch_level 3 \
#                                 --ann_positive /home/auwqh/dataset/HER2/control_group/

srun python create_patches_fp.py --source "/home/auwqh/dataset/WarwickHER2/WSI/IHC/*.ndpi" \
                                 --step_size 256 --patch_size 256 --patch --seg \
                                 --save_dir /home/auwqh/dataset/WarwickHER2/patches_features_10x/ \
                                 --patch_level 2 \
                                 --ann_positive /home/auwqh/dataset/WarwickHER2/control_group/

srun python create_patches_fp.py --source "/home/auwqh/dataset/WarwickHER2/WSI/IHC/*.ndpi" \
                                 --step_size 256 --patch_size 256 --patch --seg \
                                 --save_dir /home/auwqh/dataset/WarwickHER2/patches_features_5x/ \
                                 --patch_level 3 \
                                 --ann_positive /home/auwqh/dataset/WarwickHER2/control_group/