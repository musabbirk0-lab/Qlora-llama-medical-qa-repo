import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch




def parse_args():
p = argparse.ArgumentParser()
p.add_argument('--base_model', type=str, default='decapoda-research/llama-7b-hf')
p.add_argument('--adapter_path', type=str, default='outputs/qlora_lora')
p.add_argument('--prompt', type=str, default='What causes scurvy?')
p.add_argument('--max_new_tokens', type=int, default=256)
return p.parse_args()




def main():
args = parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', trust_remote_code=True)
# load adapter / peft
model = PeftModel.from_pretrained(model, args.adapter_path)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


prompt = f"### Instruction:\nAnswer the medical question concisely.\n\nQuestion: {args.prompt}\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors='pt').to(device)
with torch.no_grad():
out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
text = tokenizer.decode(out[0], skip_special_tokens=True)
print(text)


if __name__ == '__main__':
main()
