###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-03-08 14:48:35
### 

nohup bash scripts/train_tk_instruct.sh 2 5,6 t5-base 8 1e-4 0 kd 5 0.05 32 allenai kl fullprompt > logs/t5-base-1e-4_kd_warm0.05_r32_allenai_kl_fullprompt.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 1,4 t5-base 8 5e-5 0 kd 5 0.05 32 allenai kl fullprompt > logs/t5-base-5e-5_kd_warm0.05_r32_allenai_kl_fullprompt.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 7,8 t5-base 8 1e-4 0 kd 5 0.05 32 allenai kl fullprompt > logs/t5-base-1e-4_kd_warm0.05_r32_allenai_kl_fullprompt_noce.log 2>&1 &