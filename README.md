# Fine-tuning Gemma 270M on SQL Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qDuRqzCWARbiwKTkxNZDYir_aQIUwmzW?usp=sharing)
[![Hugging Face Dataset](https://img.shields.io/badge/Dataset-%F0%9F%A7%A0%20Ashisane%2Fgemma--sql-blue)](https://huggingface.co/datasets/Ashisane/gemma-sql)

Fine-tuning **Google Gemma 3 – 270M** model on a custom **SQL question–answer dataset**
using **LoRA (Low-Rank Adaptation)** and **QLoRA** for efficient GPU training.

This project was created as part of a personal learning journey to understand:

- How LLMs are fine-tuned from base models
- How dataset structure and hyperparameters affect results
- How to make training fully reproducible using Colab + Hugging Face

---

## Project Overview

This repository demonstrates the **complete process of domain-specific fine-tuning** of a small open-weight model — Gemma 270M — on structured SQL tasks.

The dataset contains ~1,500 examples covering:

- SQL theory (normalization, keys, transactions)
- Query writing and debugging
- Schema reasoning and conceptual understanding

The final fine-tuned model can handle simple SQL reasoning, code generation, and conceptual explanations.

---

## Motivation

The goal of this project is not to produce a perfect SQL assistant —but to **understand every part of the LLM fine-tuning pipeline**, including:

- Tokenization and quantization
- LoRA parameter injection
- Training arguments and their impact
- Dataset formatting for chat-style models
- Practical GPU constraints (tested on RTX 4050 6GB and Colab T4 GPU)

---

## Repository Structure

```
finetune-gemma-270m-sql/
│
├── Finetune_Gemma_SQL.ipynb     # Colab notebook (ready to run)
├── train.py                     # Standalone training script
├── dataset/
│   ├── gemma_clean_dataset.jsonl
│   └── README.md
├── learnings/                   # Personal notes on fine-tuning concepts
│   ├── finetuning_basics.md
│   ├── lora_explained.md
│   ├── quantization_notes.md
│   └── data_handling.md
├── requirements.txt
└── README.md                    # You are here
```

---

## Quick Start

### Run in Colab

Click the badge below to open the ready-to-use notebook:
👉 [**Open in Colab**](https://colab.research.google.com/drive/1qDuRqzCWARbiwKTkxNZDYir_aQIUwmzW?usp=sharing)

The notebook automatically:

- Installs dependencies
- Downloads the dataset
- Loads `google/gemma-3-270m-it`
- Applies LoRA adapters
- Starts fine-tuning
- Saves the final model

---

### Use Locally

```bash
git clone https://github.com/Ashisane/finetune-gemma-270m-sql.git
cd finetune-gemma-270m-sql
pip install -r requirements.txt
python train.py
```

---

## Dataset

Dataset: [**Ashisane/gemma-sql**](https://huggingface.co/datasets/Ashisane/gemma-sql)

| Property  | Value                                         |
| --------- | --------------------------------------------- |
| Records   | ~2,000                                        |
| Format    | JSONL (`text` field with Gemma chat markup) |
| License   | MIT                                           |
| Hosted on | Hugging Face Datasets                         |

Example record:

```json
{
  "text": "<start_of_turn>user\nWhat is normalization in SQL?\n<end_of_turn>\n<start_of_turn>model\nNormalization organizes data into tables to reduce redundancy.\n<end_of_turn>"
}
```

---

## Training Configuration

| Parameter             | Value                      |
| --------------------- | -------------------------- |
| Base model            | `google/gemma-3-270m-it` |
| Method                | LoRA / QLoRA               |
| Epochs                | 3                          |
| Learning rate         | 1e-4                       |
| Batch size            | 1                          |
| Gradient accumulation | 2                          |
| Max sequence length   | 512                        |
| Scheduler             | cosine                     |
| Warmup ratio          | 0.05                       |

Tested with:

- Colab T4 GPU ✅
- RTX 4050 (6GB) ✅
- CPU (slow, but works) ✅

---

## Results

- Training loss decreased steadily to ~1.2
- Model learns consistent SQL structure (`SELECT`, `JOIN`, etc.)
- Performs moderately well on theory + simple reasoning
- Struggles with multi-table joins (expected for small data)

---

## License

This project and dataset are released under the **MIT License**.
Free for educational and research use.

---

## Author

**Ashisane**

- [Hugging Face](https://huggingface.co/Ashisane)
- [GitHub](https://github.com/Ashisane)

Project created to explore end-to-end LLM fine-tuning for domain adaptation.

---

## Tags

`gemma` • `sql` • `lora` • `fine-tuning` • `instruction-tuning` • `qlora` • `huggingface` • `colab`

---

### Notes

This repository is meant as an **educational reference** — not as a benchmark model.It’s intentionally small-scale and transparent to help others **learn by building**.Future work may explore:

- Expanding dataset (v2, 5k+ examples)
- Comparing with Mistral, Phi-3, and Qwen models
- Evaluating SQL reasoning accuracy
