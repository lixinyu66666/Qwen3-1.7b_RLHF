import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from prompt_template import format_prompt

def generate_response(instruction, model, tokenizer, max_new_tokens=512):
    prompt = format_prompt(instruction, "")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = output_text.split("### Response:")[-1].strip()
    return response

base_model_name = "Qwen/Qwen3-1.7B"
lora_path = "checkpoints/qwen3_lora/checkpoint-2103"
load_4bit = True
device = "cuda" if torch.cuda.is_available() else "cpu"

if load_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

model = PeftModel.from_pretrained(model, lora_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

if __name__ == "__main__":
    instruction = input("Enter your instruction: ")
    response = generate_response(instruction, model, tokenizer)
    print(f"Response: {response}")