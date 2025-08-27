import re, os
import matplotlib.pyplot as plt
import numpy as np  

log_file = "logs/train_rm.out"
smooth = False
window_size = 20
save_dir = "results/reward_model_training_loss.png"

os.makedirs(os.path.dirname(save_dir), exist_ok=True)
with open(log_file, "r", encoding="utf-8") as f:
    logs = f.readlines()

pattern = re.compile(r"'loss': ([0-9]+\.[0-9]+)")
losses = []
steps = []

step = 0
for line in logs:
    match = pattern.search(line)
    if match:
        step += 1
        steps.append(step)
        losses.append(float(match.group(1)))

print(f"Total steps found: {len(steps)}")

plt.figure(figsize=(10, 6))
plt.plot(steps, losses, color='blue', label='Reward Model Training Loss')

plt.title("Training Loss Curve")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
plt.savefig(save_dir)