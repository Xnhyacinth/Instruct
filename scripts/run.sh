###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-03-08 14:48:35
### 

# nohup bash scripts/train_tk_instruct.sh 2 0,9 t5-base 8 5e-4 0 kd 3 0.03 32 allenai no fullprompt ffn > logs/t5-base-5e-4_kd_warm0.05_r32_allenai_fullprompt_ffn.log 2>&1 &

nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 8 5e-4 0 kd 3 0.03 32 allenai kl fullprompt ffn > logs/t5-base-5e-4_kd_warm0.05_r32_allenai_fullprompt_ffn_kl.log 2>&1 &