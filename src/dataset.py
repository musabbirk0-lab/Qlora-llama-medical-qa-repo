from datasets import load_dataset
import json
from typing import List, Dict




def load_jsonl(path: str):
with open(path, 'r', encoding='utf-8') as f:
for line in f:
yield json.loads(line)




def prepare_dataset(tokenizer, path: str, max_length: int = 512):
raw = list(load_jsonl(path))
texts = []
for item in raw:
q = item.get('question', '').strip()
ctx = item.get('context', '').strip()
# simple prompt format - you can customize
prompt = f"### Instruction:\nAnswer the medical question concisely.\n\nQuestion: {q}\n\nContext: {ctx}\n\n### Response:\n"
# expected answer: join answers
ans = item.get('answers')
if isinstance(ans, list):
answer = ans[0]
else:
answer = ans or ''
full = prompt + answer
texts.append({'prompt': prompt, 'response': answer, 'input_text': full})


# Tokenize
def tok(example):
enc = tokenizer(example['input_text'], truncation=True, max_length=max_length)
return {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask'], 'labels': enc['input_ids'].copy()}


tokenized = [tok(t) for t in texts]
return tokenized
