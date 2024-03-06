# Instruct

- It is built based on the pretrained [T5 model](https://arxiv.org/abs/1910.10683), and finetuned on our [data](https://github.com/allenai/natural-instructions).

## Requirements

Our main experiments and analysis are conducted on the following environment:

- CUDA (11.3)
- cuDNN (8.2.0.53)
- Pytorch (1.10.0)
- Transformers (4.17.0)
- DeepSpeed

You can refer to the [Dockerfile](Dockerfile) for setting up the environment and install the required python libraries by running

```bash
pip install -r requirements.txt
```

## Data

Our models are trained and evaluated on [Super-NaturalInstructions](https://github.com/allenai/natural-instructions), which can be cloned by running:

```bash
git clone git@github.com:allenai/natural-instructions.git data
```

Since Super-NaturalInstructions didn't provide an official split for the development set, in order to do evaluation during training time, you can mannualy create a `dev_tasks.txt` in the `data/splits/default` folder. We found it unclear what should be a meaningful validation set, under such cross-task generalization setting. You can use a part of the training tasks for validation, or you can set apart tasks in some categories for validation.

If you want to use the T5 code [here](https://github.com/google-research/text-to-text-transfer-transformer), you can convert the data into text2text format with [`scripts/convert_data_to_s2s.sh`](scripts/convert_data_to_s2s.sh).

## Training

A sample script for training the Tk-Instruct 3B model in our paper can be found at [`scripts/train_tk_instruct.sh`](scripts/train_tk_instruct.sh). You can run it as follows:

```bash
./scripts/train_tk_instruct.sh
```

However, if you are familiar with [Beaker](https://beaker.org/), you can refer to the [`beaker_configs/default_experiment.yaml`](beaker_configs/default_experiment.yaml) for a sample experiment config, and modifying [`src/create_exps.py`](src/create_exps.py) to easily starts a set of experiments by running:

```bash
python src/create_exps.py
```
