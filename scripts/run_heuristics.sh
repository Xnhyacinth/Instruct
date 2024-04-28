# output_dir="output/"

# echo "Copy_demo for English track"
# python src/run_heuristics.py --task_dir data/tasks --data_dir data/splits/default --max_num_instances_per_task 1 --max_num_instances_per_eval_task 100 --method copy_demo --output_dir ${output_dir}/default/copy_demo
# python src/compute_metrics.py --predictions ${output_dir}/default/copy_demo/predicted_examples.jsonl --track default

# echo "Copy_input for English track"
# python src/run_heuristics.py --task_dir data/tasks --data_dir data/splits/default --max_num_instances_per_task 0 --max_num_instances_per_eval_task 100 --method copy_input --output_dir ${output_dir}/default/copy_input
# python src/compute_metrics.py --predictions ${output_dir}/default/copy_input/predicted_examples.jsonl --track default

# echo "Copy_demo for x-lingual track"
# python src/run_heuristics.py --task_dir data/tasks --data_dir data/splits/xlingual --max_num_instances_per_task 1 --max_num_instances_per_eval_task 100 --method copy_demo --output_dir ${output_dir}/xlingual/copy_demo
# python src/compute_metrics.py --predictions ${output_dir}/xlingual/copy_demo/predicted_examples.jsonl --track xlingual

# echo "Copy_input for x-lingual track"
# python src/run_heuristics.py --task_dir data/tasks --data_dir data/splits/xlingual --max_num_instances_per_task 0 --max_num_instances_per_eval_task 100 --method copy_input --output_dir ${output_dir}/xlingual/copy_input
# python src/compute_metrics.py --predictions ${output_dir}/xlingual/copy_input/predicted_examples.jsonl --track xlingual

nohup bash scripts/train_tk_instruct_kd.sh 1 0 t5-base 1 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt SO > logs0/t5-base-1e-4_lora_warm0.05_pos0_SO.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct_kd.sh 1 6 t5-base 1 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt Translation > logs0/t5-base-1e-4_lora_warm0.05_pos0_Translation.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct_kd.sh 1 7 t5-base 1 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt ST > logs0/t5-base-1e-4_lora_warm0.05_pos0_ST.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct_kd.sh 1 8 t5-base 1 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt SDe > logs0/t5-base-1e-4_lora_warm0.05_pos0_SDe.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct_kd.sh 1 9 t5-base 1 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt NC > logs0/t5-base-1e-4_lora_warm0.05_pos0_NC.log 2>&1 &


sleep 10800
python down.py
sleep 60
nohup bash scripts/train_tk_instruct_kd.sh 1 0 t5-base 1 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt ID > logs0/t5-base-1e-4_lora_warm0.05_pos0_ID.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct_kd.sh 1 6 t5-base 1 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt SCl > logs0/t5-base-1e-4_lora_warm0.05_pos0_SCl.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct_kd.sh 1 7 t5-base 1 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt SE > logs0/t5-base-1e-4_lora_warm0.05_pos0_SE.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct_kd.sh 1 8 t5-base 1 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt PP > logs0/t5-base-1e-4_lora_warm0.05_pos0_PP.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct_kd.sh 1 9 t5-base 2 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt EG > logs0/t5-base-1e-4_lora_warm0.05_pos0_EG.log 2>&1 &

sleep 10800
python down.py
sleep 60
nohup bash scripts/train_tk_instruct_kd.sh 1 0 t5-base 1 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt DRC > logs0/t5-base-1e-4_lora_warm0.05_pos0_DRC.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct_kd.sh 1 6 t5-base 1 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt DCI > logs0/t5-base-1e-4_lora_warm0.05_pos0_DCI.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct_kd.sh 1 7 t5-base 1 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt PED > logs0/t5-base-1e-4_lora_warm0.05_pos0_PED.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct_kd.sh 1 8 t5-base 1 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt SRC > logs0/t5-base-1e-4_lora_warm0.05_pos0_SRC.log 2>&1 &
