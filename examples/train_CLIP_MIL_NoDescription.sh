#!/bin/bash
#SBATCH --job-name=train_CLIP_MIL         # 作业名称
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

#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config/no_destription/clip_mil_vit_b32.yaml" \
#                              --log_name "clip_mil_vit_b32_NoDescrip"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config/no_destription/clip_mil_vit_b32_BagPrompt.yaml" \
#                              --log_name "clip_mil_vit_b32_NoDescrip_BagPrompt"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config/no_destription/clip_mil_vit_b32_BagPrompt_CoOP.yaml" \
#                              --log_name "clip_mil_vit_b32_NoDescrip_BagPrompt_CoOP"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config/no_destription/clip_mil_vit_b32_BagPrompt_trainablePrompt.yaml" \
#                              --log_name "clip_mil_vit_b32_NoDescrip_BagPrompt_trainablePrompt"
#
#srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config/no_destription/clip_mil_vit_b32_BagPrompt_trainablePrompt_learnablePooling.yaml" \
#                              --log_name "clip_mil_vit_b32_NoDescrip_BagPrompt_trainablePrompt_learnabelPooling"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config/no_destription/clip_mil_vit_b32_BagPrompt_trainablePrompt_learnablePooling_contrast.yaml" \
                              --log_name "clip_mil_vit_b32_NoDescrip_BagPrompt_trainablePrompt_learnabelPooling_contrast"