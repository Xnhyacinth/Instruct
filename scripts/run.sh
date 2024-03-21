###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-03-08 14:48:35
### 

# nohup bash scripts/train_tk_instruct.sh 2 3,4 t5-base 8 3e-4 0 kd 8 0.08 32 allenai ce_kl fullprompt ffn nowhitening > logs/t5-base-3e-4_kd_warm0.08_r32_allenai_fullprompt_nowhitening_ffn_ce_kl.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 1,7 t5-base 8 5e-5 0 kd 8 0.08 32 allenai ce_kl fullprompt ffn nowhitening > logs/t5-base-5e-5_kd_warm0.08_r32_allenai_fullprompt_nowhitening_ffn_ce_kl.log 2>&1 &
nohup bash scripts/train_tk_instruct.sh 4 0,2,5,9 t5-base 4 1e-4 0 kd 3 0.03 32 allenai ce_kl fullprompt ffn whitening > logs/t5-base-1e-4_kd_warm0.03_r32_allenai_fullprompt_whitening_ffn_ce_kl.log 2>&1 &