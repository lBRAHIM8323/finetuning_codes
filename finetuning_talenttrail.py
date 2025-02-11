import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import json
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Define model ID
model_id = "meta-llama/Meta-Llama-3.1-8B"

# Function to load JSON dataset
def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return load_dataset('json', data_files=file_path)['train']

# Function to format data for fine-tuning
def format_prompts(example):
    instruction = example["instruction"]
    input_text = example["input"]
    
    try:
        output_text = json.dumps(example["output"], indent=2)  # Convert output to JSON format
    except Exception:
        output_text = str(example["output"])  # Ensure no formatting issues

    formatted_text = (
        f"<s>[INST] <<SYS>>\nYou are an AI Assistant trained to extract structured information from resumes and job descriptions.\n"
        f"Analyze the given input and extract details such as location, academic background, experience, and skills.\n<</SYS>>\n\n"
        f"{instruction}\n\n### Input: {input_text} [/INST] {output_text}</s>"
    )    
    return {"text": formatted_text}

# Load dataset and preprocess
json_file_path = '/home/ubuntu/Tripti/Project_TalentTrail/formatted_training_data.json'
data = load_data_from_json(json_file_path)
data = data.map(format_prompts)

# Function to get model and tokenizer
def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": "cuda:0", "transformer": "cuda:1"},
        trust_remote_code=True
    )
 
   
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = get_model_and_tokenizer(model_id)

# Apply LoRA (Low-Rank Adaptation)
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none", 
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

# Define output model directory
output_model = "llama3.18B-FineTuned-ResumeJD"

# Define training arguments
training_arguments = TrainingArguments(
    output_dir=output_model,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_steps=50,
    logging_steps=10,
    num_train_epochs=10,
    fp16=True,
    push_to_hub=False,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    warmup_ratio=0.05
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    peft_config=peft_config,
    dataset_text_field="text",
    args=training_arguments,
    tokenizer=tokenizer,
    max_seq_length=2048
)

# Start fine-tuning
trainer.train()
# Save the final model
trainer.save_model(output_model)

print("Training completed. Model saved to:", output_model)
