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
epoch=${8:-"3"}
warmup_ratio=${9:-"0.03"}
r=${10:-"16"}
allenai=${11:-"0"}
use_kl=${12:-"False"}
prompt=${13:-"0"}
ffn=${14:-"0"}
whitening=${15:-"0"}
pos=${16:-"2"}
s_pos=${17:-"10"}
custom=${18:-"0"}
hyper=${19:-"0"}
ko=${20:-"0"}
prefix=${21:-"0"}
gpt=${22:-"0"}
dataset=${23:-"0"}
do_sample=${24:-"0"}
cache="./cache"
echo epoch: ${epoch}
name=experiment_pos${pos}_pooler-${model}_lr${lr}_warm${warmup_ratio}
output_dir=output_pos${pos}_pooler/${model}_lr${lr}_warm${warmup_ratio}
extra_args="--evaluation_strategy no"
data_dir=data/splits/default
run_file=run_s2s.py
if [ "$e" == "eval" ];then
    cache="./cache_eval"
    name="${name}-eval"
    data_dir="${data_dir}_eval"
    output_dir="${output_dir}_eval"
    extra_args="--evaluation_strategy steps --do_eval --eval_steps 2500 --load_best_model_at_end True"
    echo name: ${name}
fi
if [ "$dataset" == "p3" ];then
    config_files=0
    data_dir=data/splits/1
    name="${name}_1"
    output_dir="${output_dir}_1"
fi
if [ "$tune" != "full" ];then
    name="${name}-${tune}_r${r}"
    output_dir="${output_dir}_${tune}_r${r}"
    run_file=run_s2s_${tune}.py
    extra_args="${extra_args} --r ${r}"
fi
if [ "$tune" == "kd" ];then
    t_model=output/${model}_lr5e-5
    if [ "$allenai" == "allenai" ];then
        if [ "$model" == "t5-base" ];then
            t_model=allenai/tk-instruct-base-def-pos
            if [ "$pos" == "0" ];then
                t_model=output_pos0/t5-base_lr1e-4_warm0.05
            fi
        fi
        if [ "$model" == "t5-xl" ];then
            t_model=allenai/tk-instruct-3b-def-pos
            if [ "$pos" == "0" ];then
                t_model=allenai/tk-instruct-3b-def 
            fi
        fi
        if [ "$model" == "t5-xxl" ];then
            t_model=allenai/tk-instruct-11b-def-pos
            if [ "$pos" == "0" ];then
                t_model=allenai/tk-instruct-11b-def 
            fi
        fi
        name="${name}_allenai"
        output_dir="${output_dir}_allenai"
    fi
    if [ "$use_kl" == "ce" ];then
        name="${name}_ce"
        output_dir="${output_dir}_ce"
        extra_args="${extra_args} --use_ce True"
    fi
    if [ "$use_kl" == "kl" ];then
        name="${name}_kl"
        output_dir="${output_dir}_kl"
        extra_args="${extra_args} --use_kl True"
    fi
    if [ "$use_kl" == "ce_kl" ];then
        name="${name}_ce_kl"
        output_dir="${output_dir}_ce_kl"
        extra_args="${extra_args} --use_kl True --use_ce True"
    fi
    if [ "$use_kl" == "all" ];then
        name="${name}_all"
        output_dir="${output_dir}_all"
        extra_args="${extra_args} --use_kl True --use_ce True --use_hd True --use_attn True"
    fi
    lora=hyperlora_kd
    if [ "$ffn" == "ffn" ];then
        name="${name}_ffn"
        output_dir="${output_dir}_ffn"
        lora="${lora}_ffn"
    fi
    if [ "$ko" == "ko" ];then
        name="${name}_ko"
        output_dir="${output_dir}_ko"
        lora="${lora}_ko"
    fi
    if [ "$prefix" != "0" ];then
        name="${name}_prefix${prefix}"
        output_dir="${output_dir}_prefix${prefix}"
        lora="${lora}_prefix"
        extra_args="${extra_args} --prefix_length ${prefix}"
        if [ "$gpt" == "gpt" ];then
            name="${name}_gpt"
            output_dir="${output_dir}_gpt"
            lora="${lora}_gpt"
        fi
    fi
    extra_args="${extra_args} --t_model ${t_model} --name ${lora} --temperature 3.0 --kd True --s_num_pos_examples ${s_pos}"
    if [ "$prompt" == "fullprompt" ];then
        name="${name}_${prompt}"
        output_dir="${output_dir}_${prompt}"
        extra_args="${extra_args} --prompt True"
    fi
    name="${name}_${whitening}"
    output_dir="${output_dir}_${whitening}"
    if [ "$whitening" == "whitening" ];then
        extra_args="${extra_args} --whitening True"
    fi
    if [ "$s_pos" == "10" ];then
        s_pos=${pos}
    fi
    if [ "$s_pos" != "$pos" ];then
        name="${name}_s_pos${s_pos}"
        output_dir="${output_dir}_s_pos${s_pos}"
    fi
    if [ "$custom" == "custom" ];then
        name="${name}_custom"
        output_dir="${output_dir}_custom"
        extra_args="${extra_args} --custom_model True"
    fi
    if [ "$hyper" == "hyper" ];then
        name="${name}_hyper"
        output_dir="${output_dir}_hyper"
        extra_args="${extra_args} --hyperencoder True"
    fi
fi
if [ "$do_sample" == "sample" ];then
    name="${name}_sample"
    output_dir="${output_dir}_sample"
    extra_args="${extra_args} --do_sample True"
fi
echo name: ${name}
echo run_file: ${run_file}
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
    --num_pos_examples ${pos} \
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
    --pad_to_max_length False \
    --learning_rate ${lr} \
    --num_train_epochs ${epoch} \
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
