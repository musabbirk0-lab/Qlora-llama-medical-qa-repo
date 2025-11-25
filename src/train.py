import os
import argparse
from transformers import (
AutoTokenizer,
AutoModelForCausalLM,
BitsAndBytesConfig,
Trainer,
TrainingArguments,
DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from src.dataset import prepare_dataset
import torch




def parse_args():
p = argparse.ArgumentParser()
p.add_argument('--model_name', type=str, default='decapoda-research/llama-7b-hf')
p.add_argument('--train_file', type=str, default='data/sample_med_qa.jsonl')
p.add_argument('--output_dir', type=str, default='outputs/qlora_lora')
p.add_argument('--per_device_train_batch_size', type=int, default=4)
p.add_argument('--num_train_epochs', type=int, default=3)
p.add_argument('--learning_rate', type=float, default=2e-4)
p.add_argument('--max_length', type=int, default=512)
p.add_argument('--lora_r', type=int, default=8)
p.add_argument('--lora_alpha', type=int, default=16)
p.add_argument('--lora_dropout', type=float, default=0.05)
return p.parse_args()




def main():
args = parse_args()


# BitsAndBytes quantization config for QLoRA style
bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.float16
)


print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
# Some tokenizers need pad token
if tokenizer.pad_token_id is None:
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


print('Loading model (quantized) - this may require GPU and bitsandbytes support')
model = AutoModelForCausalLM.from_pretrained(
args.model_name,
quantization_config=bnb_config,
device_map='auto',
trust_remote_code=True
)


model = prepare_model_for_kbit_training(model)


lora_config = LoraConfig(
r=args.lora_r,
lora_alpha=args.lora_alpha,
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
lora_dropout=args.lora_dropout,
main()
