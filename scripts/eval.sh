

# bash scripts/eval_tk_instruct.sh 1 9 t5-base 8 2 allenai
# bash scripts/eval_tk_instruct.sh 1 0 t5-base 64 2 allenai kd output_pos2_pooler/t5-base_lr1e-4_warm0.01_20000_kd_r32_allenai_ce_ffn_fullprompt_nowhitening_loramse/checkpoint-15000
# bash scripts/eval_tk_instruct.sh 1 0 t5-xl 32 2 allenai kd output_pos2_pooler/t5-xl_lr1e-4_warm0.02_kd_r32_allenai_ffn_fullprompt_nowhitening
# bash scripts/eval_tk_instruct.sh 1 0 t5-xxl 4 2 allenai kd output_pos2_pooler/t5-xxl_lr5e-5_warm0.02_kd_r32_allenai_ce_ffn_fullprompt_nowhitening/checkpoint-10000

# nohup bash scripts/eval_tk_instruct.sh 1 0 t5-base 16 2 nofid p3 output_p3_pos2_pooler/t5-base_lr1e-4_warm0.01> logs/a.log 2>&1 &
# sleep 20
nohup bash scripts/eval_tk_instruct.sh 1 1 t5-base 16 2 fid kd_p3 output_p3_pooler/t5-base_lr1e-4_warm0.03_kd_p3_r32_fid_ce_kl_ffn_fullprompt_nowhitening_stand_loramse > logs/b.log 2>&1 &