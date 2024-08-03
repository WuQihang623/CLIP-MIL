#!/bin/bash
#SBATCH --job-name=tsne         # 作业名称
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

wz=1
cd ../

### Warwick

#srun python plot_tsne.py --model "ABMIL" --fold 1 \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/ABMIL_clip_ViTB32_weights" \
#--csv_dir /home/auwqh/code/CLIP-MIL/data/Warwick \
#--h5_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/h5_files/" \
#--save_path "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/ABMIL_clip_ViTB32_weights/tsne.png"
#
#srun python plot_tsne.py --model "CLAM" --fold 1 \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/CLAM_MB_clip_ViTB32_weights" \
#--csv_dir /home/auwqh/code/CLIP-MIL/data/Warwick \
#--h5_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/h5_files/" \
#--save_path "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/CLAM_MB_clip_ViTB32_weights/tsne.png"

#srun python plot_tsne.py --model "TransMIL" --fold 1 \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/TransMIL_clip_ViTB32_weights" \
#--csv_dir /home/auwqh/code/CLIP-MIL/data/Warwick \
#--h5_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/h5_files/" \
#--save_path "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/TransMIL_clip_ViTB32_weights/tsne.png"
#
#srun python plot_tsne.py --model "CLIPMIL" --fold 1 \
#--config "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_group_ensemble/config.yaml" \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_group_ensemble/" \
#--csv_dir /home/auwqh/code/CLIP-MIL/data/Warwick \
#--h5_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/h5_files/" \
#--save_path "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)/clip_group_ensemble/tsne.png"


### ZJY

#srun python plot_tsne.py --model "ABMIL" --fold 5 \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/ABMIL_clip_ViTB32_weights" \
#--csv_dir /home/auwqh/code/CLIP-MIL/data/HER2_fold \
#--h5_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/h5_files/" \
#--save_path "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/ABMIL_clip_ViTB32_weights/tsne.png"
#
#srun python plot_tsne.py --model "CLAM" --fold 5 \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/CLAM_MB_clip_ViTB32_weights" \
#--csv_dir /home/auwqh/code/CLIP-MIL/data/HER2_fold \
#--h5_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/h5_files/" \
#--save_path "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/CLAM_MB_clip_ViTB32_weights/tsne.png"
#
#srun python plot_tsne.py --model "TransMIL" --fold 5 \
#--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/TransMIL_clip_ViTB32_weights" \
#--csv_dir /home/auwqh/code/CLIP-MIL/data/HER2_fold \
#--h5_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/h5_files/" \
#--save_path "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/TransMIL_clip_ViTB32_weights/tsne.png"

srun python plot_tsne.py --model "CLIPMIL" --fold 5 \
--config "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_group/config.yaml" \
--checkpoint_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_group/" \
--csv_dir /home/auwqh/code/CLIP-MIL/data/HER2_fold \
--h5_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/h5_files/" \
--save_path "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)/clip_group/tsne.png"