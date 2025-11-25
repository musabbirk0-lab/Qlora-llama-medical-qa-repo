

# QLoRA Fine-Tuning LLaMA for Medical Question Answering

This project demonstrates how to fine-tune a LLaMA-family language model using **QLoRA** (4-bit quantization with bitsandbytes) and **PEFT/LoRA** for a lightweight, GPU-efficient **medical question answering** task.
The repository includes a complete training pipeline, dataset preparation utilities, inference script, and a minimal example dataset to help you get started immediately.

---

## 🚀 Features

* **QLoRA 4-bit quantization** for low-VRAM fine-tuning
* **PEFT/LoRA adapters** to reduce trainable parameter count
* **End-to-end training pipeline** using HuggingFace Transformers + Accelerate
* **JSONL medical QA dataset format** with example file included
* **Inference script** for generating answers from the fine-tuned adapter
* **Clean repo structure** for easy understanding and GitHub presentation



## 📦 Installation

```bash
pip install -r requirements.txt
```

Requirements include:

* transformers
* peft
* bitsandbytes
* accelerate
* datasets
* torch

Make sure your PyTorch + CUDA version is compatible with bitsandbytes.

---

## 🧠 Training

To start training:

```bash
accelerate launch scripts/run_train.sh
```

or manually:

```bash
accelerate launch src/train.py \
  --model_name decapoda-research/llama-7b-hf \
  --train_file data/sample_med_qa.jsonl \
  --output_dir outputs/qlora_lora
```

This will:

* load the base LLaMA model in **4-bit mode**
* apply LoRA adapters
* fine-tune using your medical QA dataset
* save adapter weights into `outputs/`

---

## 🔍 Inference

Generate answers using the fine-tuned adapter:

```bash
python src/inference.py \
  --base_model decapoda-research/llama-7b-hf \
  --adapter_path outputs/qlora_lora \
  --prompt "What is the first-line treatment for acute asthma?"
```

---

## 📚 Dataset Format (JSONL)

Each line should contain:

```json
{
  "question": "What deficiency causes scurvy?",
  "context": "",
  "answers": ["vitamin C"]
}
```

A small example file is included at `data/sample_med_qa.jsonl`.


