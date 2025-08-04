import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from prompt_template import format_prompt
from evaluate import load as load_metric
import json
from tqdm import tqdm
import nltk

# nltk.download('wordnet')
# nltk.download('punkt')

def load_model(base_model_name, bnb_config, lora_path, load_lora=True):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    if load_lora:
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return model

def generate_response(instruction, model, tokenizer, max_new_tokens=512):
    prompt = format_prompt(instruction, "")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = output_text.split("### Response:")[-1].strip()
    return response

def evaluate_model(model, eval_data, metrics, name=""):
    predictions = []
    references = []

    for item in tqdm(eval_data, desc=f"Evaluating {name}"):
        pred = generate_response(item['instruction'], model, tokenizer)
        predictions.append([pred.strip()])
        references.append([[item['response'].strip()]])
    
    if 'bleu' in metrics:
        bleu_scores = bleu.compute(predictions=predictions, references=references)
        print(f"BLEU Score for {name}: {bleu_scores['bleu']:.4f}")
    if 'meteor' in metrics:
        meteor_scores = meteor.compute(predictions=predictions, references=references)
        print(f"METEOR Score for {name}: {meteor_scores['meteor']:.4f}")
    if 'rouge' in metrics:
        rouge_scores = rouge.compute(predictions=predictions, references=references)
        print(f"ROUGE-L Score for {name}: {rouge_scores['rougeL']:.4f}")
        print(f"ROUGE-1 Score for {name}: {rouge_scores['rouge1']:.4f}")
        print(f"ROUGE-2 Score for {name}: {rouge_scores['rouge2']:.4f}")

base_model_name = "Qwen/Qwen3-1.7B"
lora_path = "checkpoints/qwen3_lora/checkpoint-2103"
eval_data_path = "data/sharegpt_qa.json"
max_new_tokens = 512

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

with open(eval_data_path, 'r') as f:
    eval_data = [json.loads(line) for line in f][5000:6000]

bleu = load_metric("bleu")
meteor = load_metric("meteor")
rouge = load_metric("rouge")

model_finetuned = load_model(base_model_name, bnb_config, lora_path, load_lora=True)
evaluate_model(model_finetuned, eval_data, metrics=['bleu', 'meteor', 'rouge'], name="Qwen3 Finetuned Model")
model_base = load_model(base_model_name, bnb_config, lora_path, load_lora=False)
evaluate_model(model_base, eval_data, metrics=['bleu', 'meteor', 'rouge'], name="Qwen3 Base Model")