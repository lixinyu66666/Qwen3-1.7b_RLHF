# Qwen3-1.7B Reinforcement Learning from Human Feedback

A LoRA fine-tuning project for Qwen3-1.7B model using ShareGPT format data for instruction tuning.

## Project Overview

This project implements efficient fine-tuning of the Qwen3-1.7B large language model using LoRA (Low-Rank Adaptation) technology and 4-bit quantization, significantly reducing memory requirements while maintaining model performance. The project uses ShareGPT format dialogue data for instruction fine-tuning, suitable for dialogue generation, question answering, and other downstream tasks.

## Features

- **Efficient Fine-tuning**: Uses LoRA technology to fine-tune only a small number of parameters
- **Memory Optimization**: Supports 4-bit quantization to reduce memory usage
- **Instruction Tuning**: Instruction following training based on ShareGPT format data
- **Easy to Use**: Provides complete training, inference, and evaluation scripts
- **Flexible Configuration**: Supports custom LoRA parameters and training configurations

## Requirements

### Hardware Requirements
- GPU: Recommended NVIDIA GPU with at least 8GB VRAM
- System: Linux/Windows/macOS

### Software Dependencies
- Python 3.10+
- CUDA 11.8+ (if using GPU)

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/lixinyu66666/Qwen3-1.5b_finetune_sharegpt.git
cd Qwen3-1.7b_finetune_sharegpt
```

2. **Create environment**
```bash
conda env create -f environment.yml
conda activate qwen3finetune
```

3. **Install Flash Attention (optional but recommended)**
```bash
pip install flash-attn --no-build-isolation
```

## Data Preparation

The project uses ShareGPT format data, where each data entry contains:
```json
{
    "instruction": "User instruction or question",
    "response": "Model response"
}
```

Place the data in the `data/sharegpt_qa.json` file. You can use the provided preprocessing script:
```bash
python data/preprocess_sharegpt.py
```

## Usage

### Model Fine-tuning

Start training with the following command:
```bash
python scripts/train_lora.py
```

Training configurations can be modified in `configs/training_args.json`:
- `per_device_train_batch_size`: Batch size per device
- `learning_rate`: Learning rate
- `num_train_epochs`: Number of training epochs
- `save_steps`: Steps to save checkpoints

### Model Inference

After training, use the following command for inference:
```bash
python scripts/inference.py
```

### Model Evaluation

Evaluate model performance:
```bash
python scripts/eval.py
```

## Configuration

### LoRA Configuration (`configs/lora_config.json`)
```json
{
  "r": 8,
  "lora_alpha": 16,
  "target_modules": ["q_proj", "v_proj"],
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

**Parameter descriptions:**
- `r`: LoRA rank (controls the dimension of low-rank matrices)
- `lora_alpha`: LoRA scaling parameter (affects the strength of adaptation)
- `target_modules`: Target modules to apply LoRA (attention projection layers)
- `lora_dropout`: Dropout rate for LoRA layers
- `bias`: Bias setting for LoRA adaptation
- `task_type`: Task type for the model (causal language modeling)

### Training Configuration (`configs/training_args.json`)
Key parameters:
- `output_dir`: Output directory
- `per_device_train_batch_size`: Batch size
- `gradient_accumulation_steps`: Gradient accumulation steps
- `learning_rate`: Learning rate
- `fp16`: Whether to enable half-precision training

## Project Structure

```
Qwen3-1.7b_finetune_sharegpt/
├── README.md                   # Project documentation
├── LICENSE                     # License file
├── environment.yml             # Conda environment configuration
├── .gitignore                  # Git ignore file
├── configs/                    # Configuration directory
│   ├── lora_config.json       # LoRA configuration
│   └── training_args.json     # Training parameters
├── scripts/                    # Scripts directory
│   ├── train_lora.py          # LoRA training script
│   ├── inference.py           # Inference script
│   ├── eval.py                # Evaluation script
│   └── prompt_template.py     # Prompt template
├── data/                       # Data directory
│   ├── sharegpt_qa.json       # Training data
│   └── preprocess_sharegpt.py # Data preprocessing script
├── checkpoints/                # Model checkpoints directory
└── logs/                       # Training logs directory
```

## Customization

### Modify Target Modules
You can modify `target_modules` in `configs/lora_config.json` to specify which modules to fine-tune:
```json
"target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
```

### Adjust Training Parameters
Adjust parameters in `configs/training_args.json` based on your hardware configuration:
- Small GPU memory: Reduce `per_device_train_batch_size`, increase `gradient_accumulation_steps`
- Faster training: Increase `per_device_train_batch_size`, reduce `gradient_accumulation_steps`

### Modify Prompt Template
You can customize the prompt format in `scripts/prompt_template.py`.

## Performance Optimization

1. **Memory Optimization**:
   - Use 4-bit quantization (enabled by default)
   - Adjust batch size and gradient accumulation steps
   - Enable gradient checkpointing (enabled by default)

2. **Training Acceleration**:
   - Use Flash Attention
   - Enable FP16 mixed precision training
   - Use DeepSpeed (optional)

3. **Data Optimization**:
   - Set reasonable maximum sequence length
   - Preprocess data to improve loading speed

## Performance
Using 1000 Q&A questions from the sharegpt dataset as the test set, the following figure shows the performance of Qwen3-1.7b before and after fine-tuning.

<table class="triple-line" style="font-size:1em;">
  <thead>
    <tr>
      <th rowspan="2" style="text-align:left;">Models</th>
      <th colspan="1">BLEU score</th>
      <th colspan="1">METEOR score</th>
      <th colspan="1">ROUGE-L score</th>
      <th colspan="1">ROUGE-1 score</th>
      <th colspan="1">ROUGE-2 score</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="text-align:left;">Qwen3-1.7b</td>             <td>0.0322</td><td>0.1637</td><td>0.1780</td><td>0.2678</td><td>0.0944</tr>
    <tr><td style="text-align:left;">Qwen3-1.7b sharegpt finetune</td>         <td><strong>0.0432</strong></td><td><strong>0.1885</strong></td><td><strong>0.2060</strong></td><td><strong>0.3043</strong></td><td><strong>0.1198</strong></td></tr>
    
  </tbody>
</table>

It can be seen that after fine-tuning with the sharegpt database, Qwen3-1.7b's answers on the test set match the semantics of the labels more closely.
## Contributing

Welcome to submit Issues and Pull Requests to improve this project!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen) for providing excellent foundation models
- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Microsoft](https://github.com/microsoft/LoRA) for LoRA technology

## Contact

If you have any questions or suggestions, please contact us through:
- GitHub Issues: [Project Issues](https://github.com/lixinyu66666/Qwen3-1.5b_finetune_sharegpt/issues)
- Email: lixinyu020620@gmail.com

---
If this project helps you, please give it a Star!