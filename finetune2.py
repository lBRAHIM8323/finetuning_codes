import os
import json
import torch
from datasets import load_dataset
from transformers import (
    LlamaForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType

# === CONFIGURATION ===
# Path to the JSON file that contains records with 'context', 'question', and 'answer'
json_file = "/home/ubuntu/ibrahim/US/Context_qna/main_context.json"  # Update with your file path

# Model configuration
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Update with your model name

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# === LOAD MODEL AND TOKENIZER ===
model = LlamaForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the pad token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# === CONFIGURE LoRA ===
peft_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# === LOAD DATASET ===
# The JSON file should be a list of dictionaries where each dictionary has 'context', 'question', and 'answer'
dataset = load_dataset("json", data_files=json_file, split="train")

# === PREPROCESSING FUNCTION ===
def preprocess_function(example):
    # Create the prompt using context, question, and answer
    prompt = (
        f"Context: {example['context']}\n"
        f"Prompt: {example['prompt']}\n"
        f"Response: {example['response']}"
    )
    
    # Tokenize the prompt text. Here, we use padding up to a max_length.
    tokenized = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_tensors="pt"
    )
    
    # For causal language modeling, the labels are typically the same as input_ids.
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    # Convert the tensors to lists so that the dataset stores plain Python types.
    return {
        "input_ids": tokenized["input_ids"][0].tolist(),
        "attention_mask": tokenized["attention_mask"][0].tolist(),
        "labels": tokenized["labels"][0].tolist()
    }

# Apply the preprocessing to the entire dataset
processed_dataset = dataset.map(
    preprocess_function,
    remove_columns=dataset.column_names,
    desc="Tokenizing dataset"
)

# === SPLIT DATASET INTO TRAIN AND EVAL SETS ===
split_dataset = processed_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# === TRAINING ARGUMENTS ===
training_args = TrainingArguments(
    output_dir="./results_with_context",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="adamw_torch",
    logging_dir="./logs_with_context",
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=1,
    fp16=True,
    remove_unused_columns=False,
)

# === DATA COLLATOR ===
def data_collator(data):
    return {
        'input_ids': torch.stack([torch.tensor(sample['input_ids']) for sample in data]),
        'attention_mask': torch.stack([torch.tensor(sample['attention_mask']) for sample in data]),
        'labels': torch.stack([torch.tensor(sample['labels']) for sample in data])
    }

# === INITIALIZE TRAINER ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# === TRAINING ===
trainer.train()

# === SAVE THE FINE-TUNED MODEL AND TOKENIZER ===
model.save_pretrained("./fine_tuned_model_with_context")
tokenizer.save_pretrained("./fine_tuned_model_with_context")
