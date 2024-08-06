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


### Warwick numhead(4)

#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_instance.yaml" \
#                              --log_name "clip_instance" --fold 1 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_instance_sample.yaml" \
#                              --log_name "clip_instance_sample" --fold 1 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_instance_sample_ensemble.yaml" \
#                              --log_name "clip_instance_sample_ensemble" --fold 1 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_instance_ensemble.yaml" \
#                              --log_name "clip_instance_ensemble" --fold 1 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_stain.yaml" \
#                              --log_name "clip_stain" --fold 1 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_stain_sample.yaml" \
#                              --log_name "clip_stain_sample" --fold 1 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_stain_sample_ensemble.yaml" \
#                              --log_name "clip_stain_sample_ensemble" --fold 1 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_stain_ensemble.yaml" \
#                              --log_name "clip_stain_ensemble" --fold 1 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_concat.yaml" \
#                              --log_name "clip_concat" --fold 1 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_concat_sample.yaml" \
#                              --log_name "clip_concat_sample" --fold 1 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_concat_sample_ensemble.yaml" \
#                              --log_name "clip_concat_sample_ensemble" --fold 1 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"

#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_concat_ensemble.yaml" \
#                              --log_name "clip_concat_ensemble" --fold 1 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_group.yaml" \
#                              --log_name "clip_group" --fold 1 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_group_sample.yaml" \
#                              --log_name "clip_group_sample" --fold 1 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_group_sample_ensemble.yaml" \
#                              --log_name "clip_group_sample_ensemble" --fold 1 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_group_ensemble.yaml" \
#                              --log_name "clip_group_ensemble" --fold 1 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_group_ensemble_vpt1.yaml" \
                              --log_name "clip_group_ensemble_vpt1" --fold 1 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_group_ensemble_vpt10.yaml" \
                              --log_name "clip_group_ensemble_vpt10" --fold 1 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_group_ensemble_vpt100.yaml" \
                              --log_name "clip_group_ensemble_vpt100" --fold 1 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/Warwick_HER2/clip_numhead(4)" \
                              --feat_dir "/home/auwqh/dataset/WarwickHER2/patches_features_20x/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/Warwick"

#### ZJY_old  numhead(4)

#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_instance.yaml" \
#                              --log_name "clip_instance" --fold 5 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_instance_sample.yaml" \
#                              --log_name "clip_instance_sample" --fold 5 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_instance_sample_ensemble.yaml" \
#                              --log_name "clip_instance_sample_ensemble" --fold 5 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"

#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_instance_ensemble.yaml" \
#                              --log_name "clip_instance_ensemble" --fold 5 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_stain.yaml" \
#                              --log_name "clip_stain" --fold 5 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_stain_sample.yaml" \
#                              --log_name "clip_stain_sample" --fold 5 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_stain_sample_ensemble.yaml" \
#                              --log_name "clip_stain_sample_ensemble" --fold 5 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"

#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_stain_ensemble.yaml" \
#                              --log_name "clip_stain_ensemble" --fold 5 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_concat.yaml" \
#                              --log_name "clip_concat" --fold 5 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_concat_sample.yaml" \
#                              --log_name "clip_concat_sample" --fold 5 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_concat_sample_ensemble.yaml" \
#                              --log_name "clip_concat_sample_ensemble" --fold 5 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"

#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_concat_ensemble.yaml" \
#                              --log_name "clip_concat_ensemble" --fold 5 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_group.yaml" \
#                              --log_name "clip_group" --fold 5 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_group_sample.yaml" \
#                              --log_name "clip_group_sample" --fold 5 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_group_sample_ensemble.yaml" \
#                              --log_name "clip_group_sample_ensemble" --fold 5 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_group_ensemble.yaml" \
#                              --log_name "clip_group_ensemble" --fold 5 \
#                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
#                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
#                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_group_ensemble_vpt1.yaml" \
                              --log_name "clip_group_ensemble_vpt1" --fold 5 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_group_ensemble_vpt10.yaml" \
                              --log_name "clip_group_ensemble_vpt10" --fold 5 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_group_ensemble_vpt100.yaml" \
                              --log_name "clip_group_ensemble_vpt100" --fold 5 \
                              --save_dir "/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2_old/clip_numhead(4)" \
                              --feat_dir "/home/auwqh/dataset/HER2/patch/clip_ViTB32/pt_files/" \
                              --fold_dir "/home/auwqh/code/CLIP-MIL/data/HER2_fold"