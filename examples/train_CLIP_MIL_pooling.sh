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

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config/clip_meanPooling.yaml" \
                              --log_name "clip_meanPooling"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config/clip_attentionPooling.yaml" \
                              --log_name "clip_attentionPooling"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config/description/clip_group.yaml" \
                              --log_name "clip_description_group"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config/description/clip_group_CoOP.yaml" \
                              --log_name "clip_description_group_CoOP"


srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config/description/clip_similarity.yaml" \
                              --log_name "clip_descrip_similarity"


srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config/description/clip_similarity_CoOP.yaml" \
                              --log_name "clip_descrip_similarity_CoOP"




srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config/no_destription/clip_group.yaml" \
                              --log_name "clip_Nodescription_group"

srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config/no_destription/clip_group_CoOP.yaml" \
                              --log_name "clip_Nodescription_group_CoOP"


srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config/no_destription/clip_similarity.yaml" \
                              --log_name "clip_Nodescrip_similarity"


srun python train_CLIP_MIL.py --config "/home/auwqh/code/CLIP-MIL/examples/config/no_destription/clip_similarity_CoOP.yaml" \
                              --log_name "clip_Nodescrip_similarity_CoOP"