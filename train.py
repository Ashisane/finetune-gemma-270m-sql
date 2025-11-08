import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import os
import random
import numpy as np
from huggingface_hub import login

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

login(token="your_huggingface_token_here")

MODEL_ID = "google/gemma-3-270m-it"
DATA_PATH = "Ashisane/gemma-sql"
OUTPUT_DIR = "./gemma-sql-finetuned"
MAX_SEQ_LEN = 512
EPOCHS = 3
LR = 1e-4
BATCH_SIZE = 2
GRAD_ACCUM = 2
SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ------------------------------------------------------------
# Load model + tokenizer
# ------------------------------------------------------------
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
model.to(device)


# ------------------------------------------------------------
# Setup LoRA config
# ------------------------------------------------------------
print("Configuring LoRA...")
lora_cfg = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
print("Loading dataset...")
dataset = load_dataset(DATA_PATH, split="train")
print(f"Loaded {len(dataset)} samples.")


# ------------------------------------------------------------
# Training arguments
# ------------------------------------------------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_grad_norm=1.0,
    logging_steps=25,
    save_steps=250,
    fp16=torch.cuda.is_available(),
    bf16=False,
    seed=SEED,
    report_to="none",
    save_total_limit=2,
)


# ------------------------------------------------------------
# Initialize trainer
# ------------------------------------------------------------
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=lora_cfg,
    max_seq_length=MAX_SEQ_LEN,
    args=args,
)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete!")


# ------------------------------------------------------------
# Save model + tokenizer
# ------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model and tokenizer saved to {OUTPUT_DIR}")


# ------------------------------------------------------------
# Quick test generation
# ------------------------------------------------------------
print("\nRunning quick generation test:")
prompt = "What is a JOIN in SQL?"
input_ids = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    output = model.generate(**input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))