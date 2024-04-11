import logging
import random
import string
import torch
from transformers.data.data_collator import *
from t0_config import eval

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForP3:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    add_explanation: bool = False
    tk_instruct: bool = False
    text_only: bool = False
    kd: bool = False
    task_features: dict = None
    instruction_inputs: dict = None
    attention_masks: dict = None
    args: dict = None
    student_input: bool = False
    

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors
        
        def pad_tokens(lst, max_len, pad_id=None):
            # everything is padding token in the beginning
            if pad_id is None:
                pad_id = self.tokenizer.pad_token_id
                
            tensor = torch.ones(len(lst), max_len, dtype=torch.long) * pad_id
            # then fill each example into this big tensor
            for i, item in enumerate(lst):
                if len(item) > max_len:
                    tensor[i, :] = torch.LongTensor(item[:max_len])
                else:
                    tensor[i, :len(item)] = torch.LongTensor(item)
            return tensor
        
        if not self.args.whitening:
            sources, targets, options, tasks = [], [], [], []
            if self.kd:
                prefixs, instances, s_sources = [], [], []
            for instance in batch:
                # task_input = ""
                # # add the input first.
                # task_input += "Now complete the following example -\n"
                # task_input += f"Input: "
                # task_output = "\n"
                # task_output += "Output: "
                # input_prefix = self.tokenizer(task_input, add_special_tokens=False)["input_ids"]
                # output_prefix = self.tokenizer(task_output, add_special_tokens=False)["input_ids"]
                
                if 'story_cloze' in instance['Task']:
                    source = self.tokenizer(instance['Instance']['input'])["input_ids"]
                    targets.append(self.tokenizer(instance['Instance']['output'])["input_ids"])
                else:
                    source = instance["Instance"]["input_tokenized"]
                    targets.append(instance["Instance"]["output_tokenized"])
                
                if len(source) <= self.max_source_length:
                    sources.append(source)
                else:
                    sources.append(source[:self.max_source_length])

                if instance['Task'] in eval:
                    options.append(instance['Instance']['options'])
                    tasks.append(instance['Task'])
                
                if self.student_input:
                    # s_source
                    pass
                
                if self.kd:
                    # prefix
                    pass
                    
                if self.args.custom_model:
                    # instance
                    pass
                
        else:
           pass
        
        input_ids = pad_tokens(sources, max_len=self.max_source_length)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        decoder_input_ids = pad_tokens(targets, max_len=self.max_target_length)
        decoder_attention_mask = decoder_input_ids.ne(self.tokenizer.pad_token_id).long()

        return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, options
        