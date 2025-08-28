import os
import shutil
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import PPOConfig, PPOTrainer

# ----------------------------
# User configs (edit these)
# ----------------------------
BASE_MODEL = "Qwen/Qwen3-0.6B"
SFT_LORA_PATH = "checkpoints/qwen3_lora/checkpoint-2103"
REWARD_MODEL_PATH = "checkpoints/reward_model"
OUTPUT_DIR = "checkpoints/qwen3_ppo"
DATASET_NAME = "Anthropic/hh-rlhf"
DATASET_SPLIT = "train"
MAX_PROMPT_LEN = 256  # keep prompt reasonably long
EVAL_TAIL_SAMPLES = 100  # small eval set like TRL script

def extract_prompt(text: str) -> str:
    """Return everything up to and including the last '\\n\\nAssistant:'."""
    key = "\n\nAssistant:"
    idx = text.rfind(key)
    if idx == -1:
        return text
    return text[: idx + len(key)]

def prepare_dataset(tokenizer):
    """Return train_dataset and eval_dataset with only 'input_ids' (list of list[int])."""
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    # dataset = dataset.select(range(100000))  

    def to_prompt(batch):
        # Use 'chosen' as the source; prompts of chosen/rejected are identical in HH-RLHF
        prompts = [extract_prompt(x) for x in batch["chosen"]]
        toks = tokenizer(
            prompts,
            padding=False,            # IMPORTANT: keep lists of variable length here
            truncation=True,
            max_length=MAX_PROMPT_LEN,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        # Keep only 'input_ids' as python lists (not tensors)
        return {"input_ids": toks["input_ids"]}

    # Process in batched mode to be fast
    dataset = dataset.map(to_prompt, batched=True, remove_columns=dataset.column_names)

    # Split into train/eval (tail as eval, like the example)
    n = len(dataset)
    eval_start = max(0, n - EVAL_TAIL_SAMPLES)
    train_ds = dataset.select(range(eval_start))
    eval_ds = dataset.select(range(eval_start, n))
    return train_ds, eval_ds

def main():
    # Clean output dir like TRL example
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    # ----------------------------
    # Tokenizer
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left", trust_remote_code=True)
    # For Qwen we usually reuse EOS as PAD to avoid embedding resize
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----------------------------
    # Models
    # ----------------------------
    torch_dtype = torch.bfloat16

    # Policy with LoRA
    bnb_cfg = BitsAndBytesConfig(               
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    policy_backbone = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch_dtype, trust_remote_code=True, quantization_config=bnb_cfg  
    )
    # Reward model (sequence classification, scalar)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH, num_labels=1, torch_dtype=torch_dtype, trust_remote_code=True
    )

    # Value model: keep the same type as reward (works well with TRL's get_reward branch)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH, num_labels=1, torch_dtype=torch_dtype, trust_remote_code=True
    )

    peft_config = LoraConfig(
        r=1,
        lora_alpha=4,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",  # required by PEFT
        # Qwen/LLaMA-like module names; adjust if your model uses different names
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    policy=  get_peft_model(policy_backbone, peft_config)

    # Reference policy:
    # Use LoRA "null-ref": let PPOTrainer disable the adapter for reference (clean & memory-efficient).
    ref_model = None  

    # ----------------------------
    # Data
    # ----------------------------
    train_dataset, eval_dataset = prepare_dataset(tokenizer)

    # ----------------------------
    # PPO config
    # ----------------------------
    ppo_config = PPOConfig(
        reward_model_path="checkpoints/reward_model",
        model_adapter_name="default",
        ref_adapter_name="default",
        num_ppo_epochs=4,
    )

    # ----------------------------
    # Trainer
    # ----------------------------
    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_model,              # null-ref
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,                 # we already loaded LoRA; null-ref path does adapter toggling internally
    )

    # ----------------------------
    # Train
    # ----------------------------
    trainer.train()

    # ----------------------------
    # Save
    # ----------------------------
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # (Optional) quick qualitative samples
    trainer.generate_completions(sampling=True)

if __name__ == "__main__":
    main()
