'''
Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
Author: Xnhyacinth, Xnhyacinth@qq.com
Date: 2024-03-27 14:30:33
'''
# import datasets
# from promptsource.templates import DatasetTemplates
# from datasets import load_dataset

# prompts = DatasetTemplates('anli', None)

# print(prompts)
# config = datasets.DownloadConfig(resume_download=True, max_retries=100) 
# dataset = load_dataset("bigscience/P3", download_config=config)
# import pdb
# pdb.set_trace()
from thop import profile
import logging
import os
from pathlib import Path
import random
import string
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets.utils import set_progress_bar_enabled
from datasets import load_dataset, load_metric

import torch
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from model import T5LoraWrapper
from ni_collator import DataCollatorForNI
from ni_trainer import NIKDTrainer, NITrainer, DenserEvalCallback
from compute_metrics import compute_metrics, compute_grouped_metrics

set_progress_bar_enabled(False)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )
    r: Optional[int] = field(
        default=32,
        metadata={
            "help": "The lora rank of the model. If the model is not a lora model, this argument will be ignored."
        },
        )
    temperature: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "The temperature."
        },
        )
    load_hypernet_weights: str = field(
        default=None,
        metadata={"help": "Path to hypernet weights, otherwise random init."},
    )
    name: str = field(
        default=None,
        metadata={"help": "Path to hypernet weights, otherwise random init."},
    )
    alpha_kd: Optional[float] = field(
        default=0.4,
        metadata={"help": "weights of KD loss."}
    )
    use_kl: bool = field(
        default=False,
        metadata={
            "help": "Whether to use kl loss."
        },
    )
    use_ce: bool = field(
        default=False,
        metadata={
            "help": "Whether to use ce loss."
        },
    )
    use_hd: bool = field(
        default=False,
        metadata={
            "help": "Whether to use hidden states loss."
        },
    )
    use_attn: bool = field(
        default=False,
        metadata={
            "help": "Whether to use attention loss."
        },
    )
    select: bool = field(
        default=False,
        metadata={
            "help": "Whether to select layers for hd & attn loss."
        },
    )
    prompt: bool = field(
        default=False,
        metadata={
            "help": "Whether to use full prompt."
        },
    )
    kd: bool = field(
        default=False,
        metadata={
            "help": "Whether to knowledge distillation."
        },
    )
    whitening: bool = field(
        default=False,
        metadata={
            "help": "Whether to use whitening algorithm."
        },
    )
    custom_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to use concat for input."
        },
    )
    do_sample: bool = field(
        default=False,
        metadata={
            "help": "Whether to do_sample."
        },
    )
    pooling: Optional[str] = field(
        default="first_last_avg", metadata={"help": "Method for getting the instructions' features."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    task_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions tasks json files."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_num_instances_per_task: int = field(
        default=None, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=500, metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_task_definition: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to preappend task definition before the task input."}
    )
    num_pos_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    s_num_pos_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    num_neg_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context negative examples."}
    )
    add_explanation: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to add explanation for both the postive examples and negtive examples."}
    )
    tk_instruct: Optional[bool] = field(
        default=False,
        metadata={"help": "tk_instruct will train a model combining all valid instruction encodings. This will overwrite the other settings about instruction encoding."} 
    )
    
    def __post_init__(self):
        pass


@dataclass
class NITrainingArguments(Seq2SeqTrainingArguments):
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={"help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    do_demo: bool = field(default=False, metadata={"help": "Whether to run the model as a demo in the terminal."})


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, NITrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

model_cls = AutoModelForSeq2SeqLM
config_cls = AutoConfig
model_name_or_path = "allenai/tk-instruct-base-def-pos"
model_revision = 'main'
use_fast_tokenizer = True
use_auth_token = False
config = config_cls.from_pretrained(
    model_name_or_path,
    # cache_dir=cache_dir,
    revision=model_revision,
    use_auth_token=True if use_auth_token else None,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    # cache_dir=cache_dir,
    use_fast=use_fast_tokenizer,
    revision=model_revision,
    use_auth_token=True if use_auth_token else None,
)
model = model_cls.from_pretrained(
    model_name_or_path,
    from_tf=bool(".ckpt" in model_name_or_path),
    config=config,
    # cache_dir=cache_dir,
    revision=model_revision,
    use_auth_token=True if use_auth_token else None,
).cuda()
input = "In this task, you're given passages that contain mentions of names of people, places, or things. Some of these mentions refer to the same person, place, or thing.\
    Your job is to write questions that evaluate one's understanding of such references. Good questions are expected to link pronouns (she, her, him, his, their, etc.) or other \
        mentions to people, places, or things to which they may refer. Do not ask questions that can be answered correctly without understanding the paragraph or having multiple answers. \
    Avoid questions that do not link phrases referring to the same entity. For each of your questions, the answer should be one or more phrases in the paragraph, and it should be unambiguous.\
"

inputs = tokenizer(input, return_tensors='pt', padding='max_length', truncation=True, max_length=1024)
import pdb
pdb.set_trace()
# model = T5LoraWrapper(model, model_args.r, model_args.load_hypernet_weights, model_args)
# if model_args.load_hypernet_weights is not None:
#     model.load_state_dict(torch.load(model_args.load_hypernet_weights), strict=False, map_location=torch.device('cpu'))

flops, params = profile(model, inputs=inputs.to(model.device))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')

#base
# FLOPs = 103.819723776G
# Params = 222.833664M
#3b
# FLOPs = 1418.638614528G
# Params = 2783.649792M