set -x
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export TRANSFORMERS_CACHE=/home/yizhongw/.cache/huggingface
# allenai/tk-instruct-3b-def-pos
# export CUDA_VISIBLE_DEVICES=3
num_gpus=${1:-"1"}
echo "GPU counts: ${num_gpus}"
gpus=${2:-"8"}
echo "GPU: ${gpus}"
model=${3:-"t5-large"}
batch_size=${4:-"8"}
pos=${5:-"2"}
allenai=${6:-"0"}
model=google/${model}-lm-adapt

if [ "$allenai" == "allenai" ];then
    if [ "$model" == "t5-base" ];then
        model=allenai/tk-instruct-base-def-pos
    fi
    if [ "$model" == "t5-xl" ];then
        model=allenai/tk-instruct-3b-def-pos
        if [ "$pos" == "0" ];then
            model=allenai/tk-instruct-3b-def 
        fi
    fi
fi
out="output/${model}_eval_pos${pos}"
echo "model: ${model}"
echo ${out}
python src/run_s2s.py \
    --do_predict \
    --predict_with_generate \
    --evaluation_strategy "no" \
    --model_name_or_path $model \
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
    --data_dir data/splits/default \
    --task_dir data/tasks \
    --output_dir $out \
    --overwrite_output_dir \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_eval_batch_size ${batch_size}