import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from prompt_template import format_prompt
import json

def load_data(path):
    with open(path, 'r') as f:
        lines = [json.loads(line) for line in f]
    
    return [{"text": format_prompt(l['instruction'], l['response'])} for l in lines]

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

with open("configs/lora_config.json", "r") as f:
    lora_dict = json.load(f)
    lora_dict['task_type'] = TaskType[lora_dict['task_type']]
peft_config = LoraConfig(**lora_dict)
model = get_peft_model(model, peft_config)

with open("configs/training_args.json", "r") as f:
    training_args_dict = json.load(f)
training_args = TrainingArguments(**training_args_dict)

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