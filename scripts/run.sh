###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-03-08 14:48:35
### 
# nohup bash scripts/train_tk_instruct.sh 2 0,1 t5-base 16 1e-4 0 kd 4 0.04 32 allenai no fullprompt > logs/t5-base-1e-4_kd_warm0.03_r32_allenai_fullprompt.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 2,3 t5-base 16 1e-4 0 kd 6 0.05 32 allenai ce_kl fullprompt noffn nowhitening > logs/t5-base-1e-4_kd_warm0.05_r32_allenai_fullprompt_nowhitening_ce_kl.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 8,9 t5-base 8 1e-4 eval kd 3 0.03 32 allenai ce_kl fullprompt noffn nowhitening > logs/t5-base-1e-4_kd_warm0.03_r32_allenai_fullprompt_nowhitening_ce_kl_eval.log 2>&1 &

nohup bash scripts/train_tk_instruct.sh 2 5,6 t5-base 8 5e-5 0 kd 5 0.05 32 allenai no fullprompt noffn whitening > logs/t5-base-5e-5_kd_warm0.05_r32_allenai_fullprompt_whitening.log 2>&1 &