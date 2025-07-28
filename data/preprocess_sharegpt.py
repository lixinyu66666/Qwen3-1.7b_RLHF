from datasets import load_dataset
import json

# Load the ShareGPT Crystalcareai/Code-feedback-sharegpt-renamed dataset
# Adjust the split as needed, here we load the first 5000 samples for demonstration
dataset = load_dataset("Crystalcareai/Code-feedback-sharegpt-renamed", split="train[:5000]")
pairs = []

for item in dataset:
    convs = item['messages']
    for i in range(len(convs) - 1):
        if convs[i]['role'] == 'human' and convs[i + 1]['role'] == 'gpt':
            pairs.append({
                "instruction": convs[i]['value'],
                "response": convs[i + 1]['value']
            })

with open("data/sharegpt_qa.json", "w") as f:
    for pair in pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")