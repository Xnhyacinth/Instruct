#!/bin/bash
###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-03-01 04:45:45
### 
set -e
export NCCL_P2P_DISABLE=1
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export TRANSFORMERS_CACHE=/home/yizhongw/.cache/huggingface
# export HF_HOME=/mnt/publiccache/huggingface/
# export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
port=$(shuf -i25000-30000 -n1)
num_gpus=${1:-"1"}
echo "GPU counts: ${num_gpus}"
gpus=${2:-"8"}
echo "GPU: ${gpus}"
model=${3:-"t5-large"}
echo "model: ${model}"
bs=${4:-"4"}
echo "batch size: ${bs}"
lr=${5:-"5e-5"}
echo "lr: ${lr}"
e=${6:-"0"}
tune=${7:-"full"}
warmup_ratio=${8:-"0.00"}
r=${9:-"16"}
cache="./cache"
name=experiment-${model}_lr${lr}_warm${warmup_ratio}
output_dir=output/${model}_lr${lr}_warm${warmup_ratio}
extra_args="--evaluation_strategy no"
data_dir=data/splits/default
run_file=run_s2s_kd.py
if [ "$e" == "eval" ];then
    cache="./cache_eval"
    name="${name}-eval"
    data_dir="${data_dir}_eval"
    output_dir="${output_dir}_eval"
    extra_args="--evaluation_strategy steps --do_eval --eval_steps 2500"
    echo name: ${name}
fi
if [ "$tune" == "lora" ];then
    name="${name}-lora_r${r}"
    output_dir="${output_dir}_lora_r${r}"
    run_file=run_s2s_lora.py
    extra_args="${extra_args} --r ${r}"
    echo name: ${name}
fi
deepspeed --master_port $port -i localhost:${gpus} src/${run_file} \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path google/${model}-lm-adapt \
    --max_source_length 1024 \
    --max_target_length 128 \
    --generation_max_length 128 \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir ${data_dir} \
    --task_dir data/tasks \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --cache_dir ${cache} \
    --overwrite_cache \
    --per_device_train_batch_size ${bs} \
    --per_device_eval_batch_size ${bs} \
    --gradient_accumulation_steps 2 \
    --learning_rate ${lr} \
    --num_train_epochs 3 \
    --lr_scheduler_type constant \
    --warmup_ratio ${warmup_ratio} \
    --logging_strategy steps \
    --logging_steps 500 \
    --save_strategy steps \
    --save_steps 2500 \
    --deepspeed ds_configs/stage2.config \
    --bf16 \
    --run_name ${name} \
    --save_total_limit 1 \
    ${extra_args}
