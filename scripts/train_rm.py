import torch, os
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

class RewardDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        chosen = item["chosen"]
        rejected = item["rejected"]

        chosen_enc = self.tokenizer(
            chosen,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        rejected_enc = self.tokenizer(
            rejected,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(),
        }

def collate_fn(batch):
    input_ids = []
    attention_masks = []
    labels = []

    for item in batch:
        # chosen 1，rejected 0
        input_ids.append(item["chosen_input_ids"])
        attention_masks.append(item["chosen_attention_mask"])
        labels.append(1.0)

        input_ids.append(item["rejected_input_ids"])
        attention_masks.append(item["rejected_attention_mask"])
        labels.append(0.0)

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.tensor(labels, dtype=torch.float)
    }


def train_reward_model():
    # Step 1: Load tokenizer and dataset
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    dataset = dataset.select(range(50000))  
    # Step 2: Build dataset and dataloader
    reward_dataset = RewardDataset(dataset, tokenizer)
    train_loader = DataLoader(reward_dataset, batch_size=4, shuffle=True)

    # Step 3: Load Qwen3 model as reward model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Step 4: Training arguments
    training_args = TrainingArguments(
        output_dir="./reward_model",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=1e-5,
        logging_steps=50,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        report_to="tensorboard",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None
    )

    # Step 5: Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=reward_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn
    )

    # Step 6: Train and save model
    trainer.train()
    trainer.save_model("./reward_model")
    tokenizer.save_pretrained("./reward_model")

if __name__ == "__main__":
    train_reward_model()