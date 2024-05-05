# coding=utf-8
# Copyright 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""P3 Dataset."""


import json
import os
import random
import datasets
from t0_config import DATA_SPLITS_SIZES, FID_METADATA, eval
from utils import load_dataset_names

logger = datasets.logging.get_logger(__name__)

_CITATION = """
@article{wang2022benchmarking,
  title={Benchmarking Generalization via In-Context Instructions on 1,600+ Language Tasks},
  author={Wang, Yizhong and Mishra, Swaroop and Alipoormolabashi, Pegah and Kordi, Yeganeh and others},
  journal={arXiv preprint arXiv:2204.07705},
  year={2022}
}
"""

_DESCRIPTION = """
Natural-Instructions v2 is a benchmark of 1,600+ diverse language tasks and their expert-written instructions. 
It covers 70+ distinct task types, such as tagging, in-filling, and rewriting. 
These tasks are collected with contributions of NLP practitioners in the community and 
through an iterative peer review process to ensure their quality. 
"""

_URL = "https://instructions.apps.allenai.org/"

dataset_names = load_dataset_names("t0", "train")
class P3Config(datasets.BuilderConfig):
    def __init__(self, *args, dataset_list=None, task_dir=None, max_num_instances_per_task=None, max_num_instances_per_eval_task=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_dir: str = task_dir
        self.dataset_list = dataset_list
        self.max_num_instances_per_task: int = max_num_instances_per_task
        self.max_num_instances_per_eval_task: int = max_num_instances_per_eval_task


class P3(datasets.GeneratorBasedBuilder):
    """NaturalInstructions Dataset."""

    VERSION = datasets.Version("2.0.0")
    BUILDER_CONFIG_CLASS = P3Config
    BUILDER_CONFIGS = [
        P3Config(name="default", description="Default config for p3")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "Task": datasets.Value("string"),
                    "Categories": datasets.Value("string"),
                    "Examples": [{
                        "id": datasets.Value("string"),
                        "input": [datasets.Value("int32")],
                        "output": [datasets.Value("int32")],
                    }],
                    "Instance": {
                        "id": datasets.Value("string"),
                        "input_tokenized": [datasets.Value("int32")],
                        "output_tokenized": [datasets.Value("int32")],
                        "input": datasets.Value("string"),
                        "output": datasets.Value("string"),
                        "options": [datasets.Value("string")],
                    },
                    # "input": datasets.Value("string"),
                    # "output": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/allenai/natural-instructions",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None or self.config.task_dir is None:
            dl_path = dl_manager.download_and_extract(_URL)
            self.config.data_dir = self.config.data_dir or os.path.join(
                dl_path, "splits")
            self.config.task_dir = self.config.task_dir or os.path.join(
                dl_path, "tasks")

        split_dir = self.config.data_dir
        task_dir = self.config.task_dir
        dataset_list = self.config.dataset_list

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    # "path": os.path.join(split_dir, "train_tasks.txt"),
                    "tasks": dataset_list,  # dataset_list
                    "task_dir": task_dir,
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "subset": "train"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    # "path": os.path.join(split_dir, "dev_tasks.txt"),
                    "tasks": [],
                    "task_dir": task_dir,
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "dev"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    # "path": os.path.join(split_dir, "test_tasks.txt"),
                    "tasks": eval,  # eval
                    "task_dir": split_dir,
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "test"
                }),
        ]

    def _generate_examples(self, tasks=None, task_dir=None, max_num_instances_per_task=None, subset=None):
        """Yields examples."""
        logger.info(f"Generating tasks from = {tasks}")
        # with open(path, encoding="utf-8") as split_f:
        #     for line in split_f:
        #         task_name = line.strip()
        for task_name in tasks:
            task_path = os.path.join(task_dir, task_name + ".json")
            with open(task_path, encoding="utf-8") as task_f:
                s = task_f.read()
                task_data = json.loads(s)
                task_data["Task"] = task_name
                
                if "Instruction Source" in task_data:
                    task_data.pop("Instruction Source")

                all_instances = task_data.pop("Instances")
                if subset == "test":
                    # for testing tasks, 100 instances are selected for efficient evaluation and they are label-balanced.
                    # we put them in the first for reproducibility.
                    # so, we use them here
                    instances = all_instances
                    task_data['Examples'] = []
                    task_data['Categories'] = ''
                else:
                    instances = all_instances
                    for dataset_name in dataset_names:
                        if task_name.startswith(dataset_name):
                            task_data["Categories"] = dataset_name
                if max_num_instances_per_task is not None and max_num_instances_per_task >= 0:
                    random.shuffle(instances)
                    if subset == "train":
                        task_data['Examples'] = instances[-16:]
                        instances = instances[:-16]
                    instances = instances[:max_num_instances_per_task]

                for idx, instance in enumerate(instances):
                    example = task_data.copy()
                    if "id" not in instance:
                        instance["id"] = str(idx)
                    instance["id"] = str(instance["id"])
                    example["id"] = instance["id"]
                    example["Instance"] = instance
                    if subset == "train":
                        example["Instance"]["output_tokenized"] = example["Instance"]["output"]
                        example["Instance"]["input_tokenized"] = example["Instance"]["input"]
                        example["Instance"]["output"] = ''
                        example["Instance"]["input"] = ''
                    else:
                        example["Instance"].pop('task')
                    if 'options' not in example['Instance']:
                        example['Instance']['options'] = []
                    # print(example["Instance"].keys())

                    yield f"{task_name}_{idx}", example
