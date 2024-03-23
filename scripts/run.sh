###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-03-08 14:48:35
### 

nohup bash scripts/train_tk_instruct.sh 4 2,3,4,5 t5-xl 4 1e-4 0 kd 4 0.04 32 allenai ce_kl fullprompt ffn nowhitening 0 > logs/t5-xl-1e-4_kd_warm0.04_r32_no_fullprompt_nowhitening_ffn_ce_kl_pos0.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 0,1 t5-base 16 1e-4 0 full 5 0.05 32 no ce_kl fullprompt ffn nowhitening 0 > logs/t5-base-1e-4_full_warm0.05_pos0.log 2>&1 &

# nohup bash scripts/train_tk_instruct.sh 2 0,1 t5-base 16 1e-4 0 kd 6 0.05 32 no ce_kl fullprompt ffn nowhitening > logs/t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_kl.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 16 1e-4 0 kd 6 0.05 32 no ce fullprompt ffn nowhitening > logs/t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 2,3 t5-base 16 1e-4 0 kd 6 0.05 32 allenai ce fullprompt ffn nowhitening > logs/t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_allenai.log 2>&1 &

# nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 8 1e-4 0 full 3 0.03 32 > logs/t5-base-1e-4_full_warm0.03.log 2>&1 &