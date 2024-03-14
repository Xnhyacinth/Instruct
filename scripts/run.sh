###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-03-08 14:48:35
### 
# nohup bash scripts/train_tk_instruct.sh 2 5,6 t5-base 16 5e-5 0 lora 0.03 32 > logs/t5-base-5e-5_lora_warm0.03_r32.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 8 5e-5 0 lora > logs/t5-base-5e-5_lora.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 2,3 t5-base 8 5e-5 0 > logs/t5-base-5e-5.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 5 1,2,3,4,5 t5-base 16 eval lora > logs/t5-base-lora-eval.log 2>&1 &

# bash scripts/train_tk_instruct.sh 1 4 t5-base 2 5e-5 0 kd 0.03 32
nohup bash scripts/train_tk_instruct.sh 2 1,4 t5-base 8 1e-4 0 kd 0.05 16 > logs/t5-base-1e-4_kd_warm0.03_r16.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 5,6 t5-base 16 1e-4 0 kd 0.03 16 > logs/t5-base-1e-4_kd_warm0.03_r16.log 2>&1 &