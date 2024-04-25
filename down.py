'''
Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
Author: Xnhyacinth, Xnhyacinth@qq.com
Date: 2024-02-27 18:34:50
'''
import shutil
import os
from torch import tensor
from datasets import load_dataset
import transformers
import datasets
config = datasets.DownloadConfig(resume_download=True, max_retries=100)
# dataset = datasets.load_from_disk(f'dataset/Image/NQ/test')
# print(dataset['compressed_ctxs_5'][0]['compressed_prompt'][204:])
# dataset = datasets.load_dataset( "codeparrot/self-instruct-starcoder", cache_dir="./hf_cache", download_config=config)
# data = load_dataset("Xnhyacinth/Image", 'WQ', download_config=config)
# print(data)
# data.save_to_disk('dataset/Image/WQ')
# model_name = 'qinyuany/my-t0-base'
# # model_name = 'google/t5-xl-lm-adapt'
# # tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, local_files_only=False)
# # model = transformers.AutoModelForCausalLM.from_pretrained(model_name, resume_download=True, local_files_only=False)
# tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, local_files_only=False)
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

# model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, resume_download=True, local_files_only=False)

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


# import json
# with open('src/data_dict.json', 'r') as f:
#     data_dict = json.load(f)
#     data_map = data_dict['data_map']
# print(','.join(list(data_map.keys())))


def move_file_to_parent_folder(directory, filename):
    directory = "output_meta"
    # 获取指定目录下的所有文件夹
    folders = [f for f in os.listdir(
        directory) if os.path.isdir(os.path.join(directory, f))]
    print(len(folders))
    l = 0
    # 移动文件到上一级目录
    for folder in folders:
        dst_file = os.path.join(directory, folder, filename)
        if os.path.exists(dst_file):
            l += 1
        else:
            print(folder)
            # src_file = os.path.join(
            #                     directory, folder, 'output_pos2_pooler', filename)
            # dst_file = os.path.join(directory, folder, filename)
        try:
            for xx in os.listdir(f'{directory}/{folder}'):
                for xxx in os.listdir(f'{directory}/{folder}/{xx}'):
                    src_file = os.path.join(
                        directory, folder, xx, xxx, filename)
            if os.path.exists(src_file):
                shutil.move(src_file, dst_file)
                print(f"Moved {src_file} to {dst_file}")
            
        except:
            pass
    print(l)


def rm_file_to_parent_folder(directory, filename):
    filename0 = "check"
    filename1 = "pytorch"
    filename2 = "spiece"
    # 获取指定目录下的所有文件夹
    folders = [f for f in os.listdir(
        directory) if os.path.isdir(os.path.join(directory, f))]
    print(len(folders))
    # 移动文件到上一级目录
    for folder in folders:
        try:
            for xx in os.listdir(f'{directory}/{folder}'):
                for xxx in os.listdir(f'{directory}/{folder}/{xx}'):
                    for xxxx in os.listdir(f'{directory}/{folder}/{xx}/{xxx}'):
                        if filename1 in xxxx:
                            src_file = os.path.join(
                                directory, folder, xx, xxx, xxxx)
                            print(f"Removed {src_file}")
                            os.remove(src_file)
                            # os.removedirs(src_file)
                        if filename2 in xxxx:
                            src_file = os.path.join(
                                directory, folder, xx, xxx, xxxx)
                            print(f"Removed {src_file}")
                            os.remove(src_file)
                        if filename0 in xxxx:
                            src_file = os.path.join(
                                directory, folder, xx, xxx, xxxx)
                            # os.removedirs(src_file)
                            shutil.rmtree(src_file)
                            print(f"Removed {src_file}")
        except:
            pass


# 指定目录和要移动的文件名
directory = "output_meta"
filename = "param_tensors.json"
# filename = "check"
# 调用函数移动文件
move_file_to_parent_folder(directory, filename)
rm_file_to_parent_folder(directory, filename)


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
#     json.dump({"data_map": ds, "data_num": dict(sorted(data_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True))}, f, indent=4)
