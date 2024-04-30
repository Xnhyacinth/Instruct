###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-03-08 14:48:35
### 

nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-xl 2 1e-4 0 lora 15000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt QA > logs0/t5-base-1e-4_lora_warm0.05_pos2_QA.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 2 0,1 t5-xl 2 1e-4 0 lora 15000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt PE > logs0/t5-base-1e-4_lora_warm0.05_pos2_PE.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 2 8,9 t5-xl 2 1e-4 0 lora 15000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt SA > logs0/t5-base-1e-4_lora_warm0.05_pos2_SA.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-xl 2 1e-4 0 lora 15000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt Misc. > logs0/t5-base-1e-4_lora_warm0.05_pos2_Misc.log 2>&1 &
sleep 16400
python down.py
sleep 60
nohup bash scripts/train_tk_instruct.sh 2 8,9 t5-xl 2 1e-4 0 lora 15000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt QG > logs0/t5-base-1e-4_lora_warm0.05_pos2_QG.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 2 1,6 t5-xl 2 1e-4 0 lora 15000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt TLD > logs0/t5-base-1e-4_lora_warm0.05_pos2_TLD.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 2 8,9 t5-xl 2 1e-4 0 lora 15000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt TC > logs0/t5-base-1e-4_lora_warm0.05_pos2_TC.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-xl 2 1e-4 0 lora 15000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt CCl > logs0/t5-base-1e-4_lora_warm0.05_pos2_CCl.log 2>&1 &

sleep 16400
python down.py
sleep 60
nohup bash scripts/train_tk_instruct.sh 2 8,9 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt TM > logs0/t5-base-1e-4_lora_warm0.05_pos2_TM.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 2 1,6 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt IE > logs0/t5-base-1e-4_lora_warm0.05_pos2_IE.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 2 8,9 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt WCG > logs0/t5-base-1e-4_lora_warm0.05_pos2_WCG.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt TCo > logs0/t5-base-1e-4_lora_warm0.05_pos2_TCo.log 2>&1 &

sleep 16400
python down.py
sleep 60
nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt Summarization > logs0/t5-base-1e-4_lora_warm0.05_pos2_Summarization.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 2 0,1 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt DG > logs0/t5-base-1e-4_lora_warm0.05_pos2_DG.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 2 8,9 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt TtC > logs0/t5-base-1e-4_lora_warm0.05_pos2_TtC.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 2 4,5 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt QU > logs0/t5-base-1e-4_lora_warm0.05_pos2_QU.log 2>&1 &


sleep 16400
python down.py
sleep 60
nohup bash scripts/train_tk_instruct.sh 2 6,7 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt PT > logs0/t5-base-1e-4_lora_warm0.05_pos2_PT.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 2 0,1 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt LP > logs0/t5-base-1e-4_lora_warm0.05_pos2_LP.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 2 8,9 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt StC > logs0/t5-base-1e-4_lora_warm0.05_pos2_StC.log 2>&1 &


sleep 18200
python down.py
sleep 60
nohup bash scripts/train_tk_instruct.sh 1 6 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt CC > logs0/t5-base-1e-4_lora_warm0.05_pos2_CC.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 0 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt FiTB > logs0/t5-base-1e-4_lora_warm0.05_pos2_FiTB.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 8 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt TQE > logs0/t5-base-1e-4_lora_warm0.05_pos2_StC.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 7 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt Explanation > logs0/t5-base-1e-4_lora_warm0.05_pos2_Explanation.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 1 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt GC > logs0/t5-base-1e-4_lora_warm0.05_pos2_GC.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 9 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt SI > logs0/t5-base-1e-4_lora_warm0.05_pos2_SI.log 2>&1 &

sleep 18200
python down.py
sleep 60
nohup bash scripts/train_tk_instruct.sh 1 6 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt NER > logs0/t5-base-1e-4_lora_warm0.05_pos2_NER.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 0 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt WS > logs0/t5-base-1e-4_lora_warm0.05_pos2_WS.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 8 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt SD > logs0/t5-base-1e-4_lora_warm0.05_pos2_SD.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 7 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt SCo > logs0/t5-base-1e-4_lora_warm0.05_pos2_SCo.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 1 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt TS > logs0/t5-base-1e-4_lora_warm0.05_pos2_TS.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 9 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt Paraphrasing > logs0/t5-base-1e-4_lora_warm0.05_pos2_Paraphrasing.log 2>&1 &

sleep 18200
python down.py
sleep 60
nohup bash scripts/train_tk_instruct.sh 1 6 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt CtT > logs0/t5-base-1e-4_lora_warm0.05_pos2_CtT.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 0 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt GED > logs0/t5-base-1e-4_lora_warm0.05_pos2_GED.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 8 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt SP > logs0/t5-base-1e-4_lora_warm0.05_pos2_SP.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 7 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt Mathematics > logs0/t5-base-1e-4_lora_warm0.05_pos2_Mathematics.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 1 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt II > logs0/t5-base-1e-4_lora_warm0.05_pos2_II.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 6 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt AV > logs0/t5-base-1e-4_lora_warm0.05_pos2_AV.log 2>&1 &
sleep 20
sleep 18200
python down.py
# sleep 60

# sleep 20
nohup bash scripts/train_tk_instruct.sh 1 6 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt WRC > logs0/t5-base-1e-4_lora_warm0.05_pos2_WRC.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 7 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt QD > logs0/t5-base-1e-4_lora_warm0.05_pos2_QD.log 2>&1 &

sleep 14400
python down.py
sleep 30
# sleep 20
nohup bash scripts/train_tk_instruct.sh 1 7 t5-xl 2 1e-4 0 lora 10000 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt SED > logs0/t5-base-1e-4_lora_warm0.05_pos2_SED.log 2>&1 &
# sleep 20
nohup bash scripts/train_tk_instruct.sh 1 6 t5-xl 1 1e-4 0 lora 7500 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt FV > logs0/t5-base-1e-4_lora_warm0.05_pos2_FV.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 7 t5-xl 1 1e-4 0 lora 7500 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt DST > logs0/t5-base-1e-4_lora_warm0.05_pos2_DST.log 2>&1 &

sleep 12400
python down.py
sleep 30
nohup bash scripts/train_tk_instruct.sh 1 6 t5-xl 1 1e-4 0 lora 7500 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt SC > logs0/t5-base-1e-4_lora_warm0.05_pos2_SC.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 7 t5-xl 2 1e-4 0 lora 7500 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt NSD > logs0/t5-base-1e-4_lora_warm0.05_pos2_NSD.log 2>&1 &

sleep 12400
python down.py
sleep 30
nohup bash scripts/train_tk_instruct.sh 1 6 t5-xl 1 1e-4 0 lora 7500 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt PG > logs0/t5-base-1e-4_lora_warm0.05_pos2_PG.log 2>&1 &
sleep 20
nohup bash scripts/train_tk_instruct.sh 1 7 t5-xl 1 1e-4 0 lora 7500 0.01 32 allenai ce fullprompt ffn nowhitening 0 0 nocustom nohyper noko 0 nogpt ERC > logs0/t5-base-1e-4_lora_warm0.05_pos2_ERC.log 2>&1 &
# 36614
