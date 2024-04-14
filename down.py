'''
Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
Author: Xnhyacinth, Xnhyacinth@qq.com
Date: 2024-02-27 18:34:50
'''
from datasets import load_dataset
import transformers
import datasets 
config = datasets.DownloadConfig(resume_download=True, max_retries=100) 
from torch import tensor
# dataset = datasets.load_from_disk(f'dataset/Image/NQ/test')
# print(dataset['compressed_ctxs_5'][0]['compressed_prompt'][204:])
# dataset = datasets.load_dataset( "codeparrot/self-instruct-starcoder", cache_dir="./hf_cache", download_config=config)
# data = load_dataset("Xnhyacinth/Image", 'WQ', download_config=config)
# print(data)
# data.save_to_disk('dataset/Image/WQ')
model_name = 'qinyuany/my-t0-base'
# model_name = 'google/t5-xl-lm-adapt'
# tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, local_files_only=False)
# model = transformers.AutoModelForCausalLM.from_pretrained(model_name, resume_download=True, local_files_only=False)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, local_files_only=False)
# sequence = "who are you?"
# print("Original sequence: ",sequence)
# tokenized_sequence = tokenizer.tokenize(sequence)
# print("Tokenized sequence: ",tokenized_sequence)
# encodings = tokenizer(sequence, return_tensors="pt")
# print(encodings)
# encoded_sequence = encodings['input_ids']
# print("Encoded sequence: ", encoded_sequence)
# decoded_encodings=tokenizer.decode([784, 25946, 6, 3, 4271, 6, 10635, 6, 6862, 6, 14105, 3, 3840, 908, 1])
# print("Decoded sequence: ", decoded_encodings)

model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, resume_download=True, local_files_only=False)

# print(model.encoder(**encodings))
# from huggingface_hub import snapshot_download
# import transformers
# # snapshot_download(repo_id='google/flan-t5-base',
# #                   repo_type='model',
# #                   local_dir='./models/flan-t5-base',
# #                   resume_download=True)
# a = transformers.AutoModelForSeq2SeqLM.from_pretrained(
#             't5-large', resume_download=True
#         )
# print(a.config.d_model)



# import os
# import json
# from collections import defaultdict
# path = 'data/splits/default/train_tasks.txt'
# task_dir = 'data/tasks'
# data_dict = defaultdict(int)
# with open(path, encoding="utf-8") as split_f:
#     for line in split_f:
#         task_name = line.strip()
#         task_path = os.path.join(task_dir, task_name + ".json")
#         with open(task_path, encoding="utf-8") as task_f:
#             s = task_f.read()
#             task_data = json.loads(s)
#             data_dict[task_data['Categories'][0]] += 1
# # path = 'data/splits/default/test_tasks.txt'
# # with open(path, encoding="utf-8") as split_f:
# #     for line in split_f:
# #         task_name = line.strip()
# #         task_path = os.path.join(task_dir, task_name + ".json")
# #         with open(task_path, encoding="utf-8") as task_f:
# #             s = task_f.read()
# #             task_data = json.loads(s)
# #             data_dict.append(task_data['Categories'])
# # data_dict = list(set(data_dict))

# ds = {}
# for d in data_dict.keys():
#     dd = d.split(' ')
#     ddd = ''
#     for x in dd:
#         ddd += x[0]
#     if len(dd) > 1:
#         ds[ddd] = d
#     else:
#         ds[d] = d
# with open('src/data_dict.json', 'w') as f:
#     json.dump({"data_map": ds, "data_num": data_dict}, f, indent=4)
