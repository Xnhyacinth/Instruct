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
model_name = 'google/t5-xl-lm-adapt'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, local_files_only=False)
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, resume_download=True, local_files_only=False)

model_name = 'allenai/tk-instruct-3b-def'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, local_files_only=False)
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, resume_download=True, local_files_only=False)

gpt_tokenizer = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2", max_length=1e5)
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