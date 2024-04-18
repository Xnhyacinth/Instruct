###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-03-08 14:48:35
### 
# 'QA,QG,SA,TLD,PE,Misc.'
# nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 8 1e-4 0 kd 15000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt 'QAx1' loramse > logs/t5-base-1e-4_lora_warm0.05_pos2_QAx1_loramse.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 0,1 t5-base 8 1e-4 0 kd 20000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt 'QAxx' loramse > logs/t5-base-1e-4_lora_warm0.05_pos2_QAxx_loramse.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 2,3 t5-base 8 1e-4 0 kd 20000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt 'QAxx' > logs/t5-base-1e-4_lora_warm0.05_pos2_QAxx_allenai.log 2>&1 &
nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-base 4 1e-4 0 full 15000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt QAa > logs/t5-base-1e-4_full_warm0.05_pos2_QAa.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-base 4 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt SP > logs/t5-base-1e-4_lora_warm0.05_pos2_SP.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 0,1 t5-base 8 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt StC > logs/t5-base-1e-4_lora_warm0.05_pos2_StC_allenai.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 2,3 t5-base 8 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt TQE > logs/t5-base-1e-4_lora_warm0.05_pos2_TQE_allenai.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-base 8 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt SD > logs/t5-base-1e-4_lora_warm0.05_pos2_SD.log 2>&1 &

# sleep 9200
# nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 8 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt SCo > logs/t5-base-1e-4_lora_warm0.05_pos2_SCo.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 0,1 t5-base 8 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt NSD > logs/t5-base-1e-4_lora_warm0.05_pos2_NSD_allenai.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 2,3 t5-base 8 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt GC > logs/t5-base-1e-4_lora_warm0.05_pos2_GC_allenai.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-base 8 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt WRC > logs/t5-base-1e-4_lora_warm0.05_pos2_WRC.log 2>&1 &
# echo '1111111'
# sleep 21500
# nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 8 1e-4 0 lora 15000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt TLD > logs/t5-base-1e-4_lora_warm0.05_pos2_TLD.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 0,1 t5-base 8 1e-4 0 lora 15000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt TLD > logs/t5-base-1e-4_lora_warm0.05_pos2_TLD_allenai.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 2,3 t5-base 8 1e-4 0 lora 15000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt 'Misc.' > logs/t5-base-1e-4_lora_warm0.05_pos2_Misc_allenai.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-base 8 1e-4 0 lora 15000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt 'Misc.' > logs/t5-base-1e-4_lora_warm0.05_pos2_Misc.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 2,3 t5-base 8 1e-4 0 lora 15000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt Translation > logs/t5-base-1e-4_lora_warm0.05_pos2_Translation.log 2>&1 &

# 145
# nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 8 1e-3 0 full 50000 0.05 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt p3 > logs/p3_t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_pos2.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 8 5e-5 0 kd 2 0.02 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom 0 > logs/t5-base-5e-5_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_pos2.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-base 8 1e-3 0 kd 25000 0.02 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom 0 > logs/t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_kl_pos2.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 1,3 t5-base 8 1e-3 0 kd 25000 0.02 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom 0 > logs/t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_pos0.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 8 1e-4 0 kd 4 0.03 32 allenai ce_kl fullprompt ffn nowhitening 2 2 custom > logs/t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_pos0_custom.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 0,1 t5-base 16 1e-4 0 full 5 0.05 32 no ce_kl fullprompt ffn nowhitening 0 > logs/t5-base-1e-4_full_warm0.05_pos0.log 2>&1 &