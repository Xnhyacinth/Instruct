###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-03-08 14:48:35
### 
# 'QA,QG,SA,TLD,PE,Misc.'

# nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 8 5e-5 0 kd 30000 0.01 32 allenai ce_kl fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt 0 loramse stand > logs/t5-base-5e-5_ce_kl_warm0.05_pos2_all_loramse_stand.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-base 8 5e-5 0 kd 30000 0.01 32 allenai kl fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt 0 loramse > logs/t5-base-5e-5_kl_warm0.05_pos2_all_loramse.log 2>&1 &


# nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 8 1e-4 0 kd 15000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt 'QAx1' loramse > logs/t5-base-1e-4_lora_warm0.05_pos2_QAx1_loramse.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 0,1 t5-base 8 1e-4 0 kd 20000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt 0 loramse > logs/t5-base-1e-4_lora_warm0.05_pos2_all_loramse.log 2>&1 &

# nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-base 8 1e-4 0 kd 30000 0.01 32 allenai ce_kl fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt 0 noloramse > logs/t5-base-1e-4_ce_kl_warm0.05_pos2_all.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 0,1 t5-base 8 1e-4 0 kd 30000 0.01 32 allenai kl fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt 0 loramse > logs/t5-base-1e-4_kl_warm0.05_pos2_all_loramse.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 2,3 t5-base 8 1e-4 0 kd 30000 0.01 32 allenai kl fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt 0 noloramse stand > logs/t5-base-1e-4_kl_warm0.05_pos2_all_stand.log 2>&1 &
# sleep 20

# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-base 8 1e-4 0 kd 20000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt QAa loramse > logs/t5-base-1e-4_lora_warm0.05_pos2_QAa_loramse_allenai.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-base 4 1e-4 0 full 15000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt QAa loramse > logs/t5-base-1e-4_full_warm0.05_pos2_QAa_loramse.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-base 4 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt SP > logs/t5-base-1e-4_lora_warm0.05_pos2_SP.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 0,1 t5-base 8 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt StC > logs/t5-base-1e-4_lora_warm0.05_pos2_StC_allenai.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 2,3 t5-base 8 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt TQE > logs/t5-base-1e-4_lora_warm0.05_pos2_TQE_allenai.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-base 8 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt SD > logs/t5-base-1e-4_lora_warm0.05_pos2_SD.log 2>&1 &

# sleep 9200
# nohup bash scripts/train_tk_instruct.sh 1 4 t5-base 4 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt Paraphrasing > logs/t5-base-1e-4_lora_warm0.05_pos2_Paraphrasing.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 1 0 t5-base 4 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt Mathematics > logs/t5-base-1e-4_lora_warm0.05_pos2_Mathematics_allenai.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 1 1 t5-base 1 1e-4 0 lora 2500 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt SRC > logs/t5-base-1e-4_lora_warm0.05_pos2_SRC_allenai.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 1 6 t5-base 4 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt DST > logs/t5-base-1e-4_lora_warm0.05_pos2_DST.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 1 5 t5-base 4 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt CtT > logs/t5-base-1e-4_lora_warm0.05_pos2_CtT.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 1 1 t5-base 4 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt SO > logs/t5-base-1e-4_lora_warm0.05_pos2_SO_allenai.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 1 3 t5-base 4 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt FV > logs/t5-base-1e-4_lora_warm0.05_pos2_FV_allenai.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 1 7 t5-base 4 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt AV > logs/t5-base-1e-4_lora_warm0.05_pos2_AV.log 2>&1 &
# echo '1111111'
# sleep 11200
# nohup bash scripts/train_tk_instruct.sh 1 4 t5-base 1 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt SCl > logs/t5-base-1e-4_lora_warm0.05_pos2_SCl.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 1 5 t5-base 1 1e-4 0 lora 5000 0.01 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt SDe > logs/t5-base-1e-4_lora_warm0.05_pos2_SDe.log 2>&1 &

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
nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 4 1e-4 0 kd_p3 30000 0.03 32 fid ce_kl fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt 0 nolora 0 p3 > logs/p3_t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_kl_pos2.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 8 5e-5 0 kd 2 0.02 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom 0 > logs/t5-base-5e-5_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_pos2.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-base 8 1e-3 0 kd 25000 0.02 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom 0 > logs/t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_kl_pos2.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 1,3 t5-base 8 1e-3 0 kd 25000 0.02 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom 0 > logs/t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_pos0.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 8 1e-4 0 kd 4 0.03 32 allenai ce_kl fullprompt ffn nowhitening 2 2 custom > logs/t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_pos0_custom.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 0,1 t5-base 16 1e-4 0 full 5 0.05 32 no ce_kl fullprompt ffn nowhitening 0 > logs/t5-base-1e-4_full_warm0.05_pos0.log 2>&1 &