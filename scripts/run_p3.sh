###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-03-08 14:48:35
### 
# 'QA,QG,SA,TLD,PE,Misc.'
# sleep 21500
# 23000
# nohup bash scripts/train_tk_instruct.sh 1 6 t5-base 8 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt glue_mrpc nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_glue_mrpc.log 2>&1 &

# 60000
# nohup bash scripts/train_tk_instruct.sh 1 7 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt glue_qqp nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_glue_qqp.log 2>&1 &
		
# 60000
# nohup bash scripts/train_tk_instruct.sh 1 0 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt paws nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_paws.log 2>&1 &
# sleep 20
# # 60000
# nohup bash scripts/train_tk_instruct.sh 1 1 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt kilt_tasks_hotpotqa nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_kilt_tasks_hotpotqa.log 2>&1 &
# sleep 20
# # 60000
# nohup bash scripts/train_tk_instruct.sh 1 2 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt wiki_qa nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_wiki_qa.log 2>&1 &
# sleep 20
# # 60000
# nohup bash scripts/train_tk_instruct.sh 1 3 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt adversarial_qa_dbert nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_adversarial_qa_dbert.log 2>&1 &

		
sleep 11600
# 23000
nohup bash scripts/train_tk_instruct.sh 1 6 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt adversarial_qa_dbidaf nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_adversarial_qa_dbidaf.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 7 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt adversarial_qa_droberta nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_adversarial_qa_droberta.log 2>&1 &
		
# 60000
nohup bash scripts/train_tk_instruct.sh 1 0 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt duorc_SelfRC nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_duorc_SelfRC.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 1 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt duorc_ParaphraseRC nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_duorc_ParaphraseRC.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 2 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt ropes nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_ropes.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 3 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt quoref nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_quoref.log 2>&1 &


sleep 12100
# 23000
nohup bash scripts/train_tk_instruct.sh 1 6 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt cos_e_v1.11 nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_cos_e_v1.11.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 7 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt cosmos_qa nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_cosmos_qa.log 2>&1 &
		
# 60000
nohup bash scripts/train_tk_instruct.sh 1 0 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt dream nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_dream.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 1 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt qasc nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_qasc.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 2 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt quail nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_quail.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 3 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt quartz nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_quartz.log 2>&1 &

		
sleep 12100
# 23000
nohup bash scripts/train_tk_instruct.sh 1 6 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt sciq nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_sciq.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 7 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt social_i_qa nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_social_i_qa.log 2>&1 &
		
# 60000
nohup bash scripts/train_tk_instruct.sh 1 0 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt wiki_hop_original nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_wiki_hop_original.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 1 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt wiqa nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_wiqa.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 2 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt amazon_polarity nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_amazon_polarity.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 3 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt app_reviews nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_app_reviews.log 2>&1 &

sleep 12100
# 23000
nohup bash scripts/train_tk_instruct.sh 1 6 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt imdb nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_imdb.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 7 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt rotten_tomatoes nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_rotten_tomatoes.log 2>&1 &
		
# 60000
nohup bash scripts/train_tk_instruct.sh 1 0 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt yelp_review_full nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_yelp_review_full.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 1 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt common_gen nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_common_gen.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 2 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt wiki_bio nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_wiki_bio.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 3 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt cnn_dailymail_3.0.0 nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_cnn_dailymail_3.0.0.log 2>&1 &


sleep 12100
# 23000
nohup bash scripts/train_tk_instruct.sh 1 6 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt gigaword nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_gigaword.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 7 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt multi_news nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_multi_news.log 2>&1 &
		
# 60000
nohup bash scripts/train_tk_instruct.sh 1 0 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt samsum nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_samsum.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 1 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt xsum nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_xsum.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 2 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt ag_news nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_ag_news.log 2>&1 &
sleep 20
# 60000
nohup bash scripts/train_tk_instruct.sh 1 3 t5-base 16 1e-4 0 lora_p3 10000 0.01 32 noallenai ce fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt dbpedia_14 nolora 0 p3 > logs_p3/t5-base-1e-4_lora_warm0.05_pos2_dbpedia_14.log 2>&1 &


# 145
# nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 4 1e-4 0 kd_p3 50000 0.03 32 fid ce_kl fullprompt ffn nowhitening 2 2 nocustom nohyper noko 0 nogpt 0 nolora 0 p3 > logs/p3_t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_kl_pos2.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 8 5e-5 0 kd 2 0.02 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom 0 > logs/t5-base-5e-5_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_pos2.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-base 8 1e-3 0 kd 25000 0.02 32 allenai ce fullprompt ffn nowhitening 2 2 nocustom 0 > logs/t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_kl_pos2.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 1,3 t5-base 8 1e-3 0 kd 25000 0.02 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom 0 > logs/t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_pos0.log 2>&1 &
# sleep 20
# nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-base 8 1e-4 0 kd 4 0.03 32 allenai ce_kl fullprompt ffn nowhitening 2 2 custom > logs/t5-base-1e-4_kd_warm0.05_r32_no_fullprompt_nowhitening_ffn_ce_pos0_custom.log 2>&1 &
# nohup bash scripts/train_tk_instruct.sh 2 0,1 t5-base 16 1e-4 0 full 5 0.05 32 no ce_kl fullprompt ffn nowhitening 0 > logs/t5-base-1e-4_full_warm0.05_pos0.log 2>&1 &