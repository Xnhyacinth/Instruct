# nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-base 8 5e-5 0 lora > logs/t5-base-5e-5_lora_warm0.05.log 2>&1 &
nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-base 8 5e-4 0 lora 0.03 32 > logs/t5-base-5e-4_lora_warm0.03_r32.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 8 5e-5 0 lora > logs/t5-base-5e-5_lora.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 2,3 t5-base 8 5e-5 0 > logs/t5-base-5e-5.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 5 1,2,3,4,5 t5-base 16 eval lora > logs/t5-base-lora-eval.log 2>&1 &