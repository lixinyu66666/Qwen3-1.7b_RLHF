import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import json

def load_data(path):
    with open(path, 'r') as f:
        lines = [json.loads(line) for line in f]
    
    return [{"text": f"### Instruction:\n{l['instruction']}\n\n### Response:\n{l['response']}"} for l in lines]

def tokenize(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load the dataset
dataset = load_data("data/sharegpt_qa.json")
dataset = Dataset.from_list(dataset)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize)

# LoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

current_device = torch.cuda.current_device()
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="checkpoints/qwen3_lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_dir="logs",
    save_steps=500,
    logging_steps=50,
    fp16=True,
    save_total_limit=2,
    report_to="none",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    preprocess_logits_for_metrics=None,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()