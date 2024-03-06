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
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType, PeftConfig, PeftModel
model_name_or_path = 'google/t5-base-lm-adapt'
config = AutoConfig.from_pretrained(
    model_name_or_path,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    # cache_dir=model_args.cache_dir,
)
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name_or_path,
    config=config,
    # cache_dir=model_args.cache_dir,
)
gen_kwargs = {
        "max_length": 64,
        "num_beams": 4,
    }
model = get_peft_model(model, lora_config)
model.resize_token_embeddings(len(tokenizer))
print(model.encoder.main_input_name)
print(hasattr(model, "encoder"))
input = tokenizer("translate English to German: Hugging Face is a technology company based in New York and Paris", return_tensors="pt")
print(input)
generated_tokens = model.generate(
        **input,
        **gen_kwargs,
        )