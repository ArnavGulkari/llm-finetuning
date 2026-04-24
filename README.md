# LLM Fine-Tuning with Mistral, QLoRA and PEFT

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

Fine-tuning large language models including Llama 2 and Mistral 7B 
using QLoRA and PEFT techniques for parameter-efficient training 
on consumer hardware.

---

## Models Fine-Tuned

| Model | Technique | Notebook |
|---|---|---|
| Mistral 7B | QLoRA + PEFT | `Fine_Tuning_with_Mistral_QLora_PEFt.ipynb` |
| Llama 2 | LoRA | `Fine_tune_Llama_2.ipynb` |
| General LLM | Fine-Tuning | `Fine_Tuning_LLm_Models.ipynb` |
| Custom | LoRA Tuning | `lora_tuning.ipynb` |

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Framework | PyTorch |
| Models | Mistral 7B, Llama 2 |
| Fine-Tuning | QLoRA, PEFT, LoRA |
| Library | Hugging Face Transformers |
| Environment | Google Colab / GPU |

---

## How It Works

### Step 1 — Setup Environment
```bash
pip install transformers peft bitsandbytes accelerate datasets
```

### Step 2 — Prepare Data
Preprocess your dataset into instruction format suitable 
for supervised fine-tuning.

### Step 3 — Configure Parameters
Set up fine-tuning parameters:
- Learning rate
- Batch size
- Number of epochs
- QLoRA rank and alpha values

### Step 4 — Fine-Tune
Run the fine-tuning process using Mistral with 
QLoRA and PEFT configurations.

### Step 5 — Evaluate
Evaluate model performance on validation set 
after fine-tuning.

### Step 6 — Deploy
Deploy the fine-tuned model for inference.

---

## Key Concepts

**QLoRA** — Quantized Low Rank Adaptation. Reduces GPU 
memory usage by quantizing the base model to 4-bit 
while training LoRA adapters in full precision.

**PEFT** — Parameter Efficient Fine-Tuning. Fine-tunes 
only a small subset of model parameters instead of 
the entire model — making it feasible on consumer hardware.

**LoRA** — Low Rank Adaptation. Injects trainable rank 
decomposition matrices into transformer layers.

---

## Results

- Significant reduction in GPU memory usage vs full fine-tuning
- Maintained model performance on downstream NLP tasks
- Successfully fine-tuned billion-parameter models on single GPU

---

## What I Learned

- Parameter-efficient fine-tuning of large language models
- QLoRA and PEFT techniques for memory-efficient training
- Working with Hugging Face Transformers and PEFT library
- Applying LoRA adapters to transformer attention layers
- Evaluating and deploying fine-tuned LLMs

---

## License

MIT License
