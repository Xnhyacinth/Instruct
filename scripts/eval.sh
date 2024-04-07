

# bash scripts/eval_tk_instruct.sh 1 9 t5-base 8 2 allenai
bash scripts/eval_tk_instruct.sh 1 0 t5-base 64 2 allenai kd output_pos2_pooler/t5-base_lr1e-4_warm0.03_kd_r32_allenai_ce_ffn_fullprompt_nowhitening
# bash scripts/eval_tk_instruct.sh 1 0 t5-xxl 4 2 allenai kd output_pos2_pooler/t5-xxl_lr5e-5_warm0.02_kd_r32_allenai_ce_ffn_fullprompt_nowhitening/checkpoint-10000