###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-03-08 14:48:35
### 

# 145
nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 8 1e-4 0 kd 4 0.03 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom 0 ko 32 > logs/t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_pos2.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 2 0,6 t5-base 8 5e-5 0 kd 4 0.03 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom 0 ko 32 > logs/t5-base-5e-5_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_pos2.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 2 1,7 t5-base 8 1e-4 0 kd 4 0.03 32 allenai ce_kl fullprompt ffn nowhitening 2 2 nocustom 0 ko 32 > logs/t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_kl_pos2.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 2 2,3 t5-base 8 1e-4 0 kd 4 0.03 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom 0 ko 32 > logs/t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_pos0.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 8 1e-4 0 kd 4 0.03 32 allenai ce_kl fullprompt ffn nowhitening 2 2 custom > logs/t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_pos0_custom.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 0,1 t5-base 16 1e-4 0 full 5 0.05 32 no ce_kl fullprompt ffn nowhitening 0 > logs/t5-base-1e-4_full_warm0.05_pos0.log 2>&1 &