# ============================================================
# COMPLETE QWEN 2.5 VL 3B FINE-TUNING - ROBUST VERSION
# Thai Handwriting Dataset - Kaggle GPU 16GB
# WITH FLASH-ATTN CUDA FIX
# ============================================================

import subprocess
import sys
import os
import warnings

warnings.filterwarnings('ignore')

print("="*60)
print("QWEN 2.5 VL 3B FINE-TUNING SETUP")
print("="*60)
print()

# ════════════════════════════════════════════════════════════
# CELL 1: INSTALL AND FIX DEPENDENCIES
# ════════════════════════════════════════════════════════════

print("🔧 Installing and fixing dependencies...\n")

# First, uninstall any broken flash-attn
print("[1/5] Cleaning up old packages...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "flash-attn", "flash-attention"])
except:
    pass
print("✓ Old packages removed\n")

# Clear pip cache
print("[2/5] Clearing pip cache...")
subprocess.check_call([sys.executable, "-m", "pip", "cache", "purge"])
print("✓ Cache cleared\n")

# Install flash-attn with specific version
print("[3/5] Installing flash-attn...")
try:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--no-build-isolation",
        "--force-reinstall",
        "flash-attn==2.6.3"
    ])
    print("✓ flash-attn installed\n")
    flash_attn_available = True
except Exception as e:
    print(f"⚠️  Flash-attn install failed: {str(e)[:50]}")
    print("   Will use eager attention (slower but works)\n")
    flash_attn_available = False

# Install other dependencies
print("[4/5] Installing other dependencies...")
packages = [
    "git+https://github.com/huggingface/transformers",
    "accelerate",
    "peft",
    "datasets",
    "huggingface-hub",
    "qwen-vl-utils",
]

for pkg in packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    except:
        print(f"  Warning: Could not install {pkg}, skipping...")

print("✓ Dependencies installed\n")

# Verify setup
print("[5/5] Verifying setup...")
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Try to import flash_attn
try:
    import flash_attn as fa
    print(f"  Flash-attn: {fa.__version__} ✓")
    flash_attn_available = True
except Exception as e:
    print(f"  Flash-attn: Not available - {str(e)[:30]}")
    print(f"  → Will use eager attention mode")
    flash_attn_available = False

print("\n✅ Setup complete!\n")

# ════════════════════════════════════════════════════════════
# CELL 2: IMPORT LIBRARIES
# ════════════════════════════════════════════════════════════

import torch
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info

print("✓ All libraries imported\n")

# ════════════════════════════════════════════════════════════
# CELL 3: CONFIGURATION
# ════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    MODEL_NAME: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    DATASET_NAME: str = "iapp/thai_handwriting_dataset"
    MAX_LENGTH: int = 8192
    LORA_R: int = 64
    LORA_ALPHA: int = 16
    LORA_DROPOUT: float = 0.05
    LORA_TARGET_MODULES: list = None
    BATCH_SIZE: int = 1
    GRADIENT_ACCUMULATION_STEPS: int = 4
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 3
    WARMUP_STEPS: int = 100
    MIN_PIXELS: int = 256 * 28 * 28
    MAX_PIXELS: int = 1280 * 28 * 28
    OUTPUT_DIR: str = "./qwen_finetuned_thai"
    USE_FLASH_ATTENTION_2: bool = flash_attn_available  # Auto-detect

    def __post_init__(self):
        if self.LORA_TARGET_MODULES is None:
            self.LORA_TARGET_MODULES = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

config = TrainingConfig()

print(f"Configuration:")
print(f"  Model: {config.MODEL_NAME}")
print(f"  Attention: {'Flash-Attn 2' if config.USE_FLASH_ATTENTION_2 else 'Eager'}")
print(f"  Epochs: {config.NUM_EPOCHS}")
print(f"  Batch size: {config.BATCH_SIZE}\n")

# ════════════════════════════════════════════════════════════
# CELL 4: LOAD MODEL AND PROCESSOR
# ════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model (may take 2-3 minutes)...\n")
print(f"Device: {device}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

attn_impl = "flash_attention_2" if config.USE_FLASH_ATTENTION_2 else "eager"

try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"✓ Loaded with {attn_impl} attention")
except Exception as e:
    print(f"⚠️  {attn_impl} failed, trying eager attention")
    print(f"   Error: {str(e)[:60]}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"✓ Loaded with eager attention")

processor = Qwen2_5_VLProcessor.from_pretrained(
    config.MODEL_NAME,
    min_pixels=config.MIN_PIXELS,
    max_pixels=config.MAX_PIXELS,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    config.MODEL_NAME,
    trust_remote_code=True,
    use_fast=False,
)

model.enable_input_require_grads()
model.gradient_checkpointing_enable()

print(f"✓ Model: {config.MODEL_NAME}")
print(f"✓ Parameters: {model.num_parameters():,}\n")

# ════════════════════════════════════════════════════════════
# CELL 5: SETUP LoRA
# ════════════════════════════════════════════════════════════

print("Setting up LoRA...\n")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=config.LORA_R,
    lora_alpha=config.LORA_ALPHA,
    lora_dropout=config.LORA_DROPOUT,
    target_modules=config.LORA_TARGET_MODULES,
    bias="none",
    inference_mode=False,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
print()

# ════════════════════════════════════════════════════════════
# CELL 6: LOAD DATASET
# ════════════════════════════════════════════════════════════

print("Loading dataset...\n")

dataset = load_dataset(config.DATASET_NAME, split='train')
print(f"Total available: {len(dataset)} samples\n")

# ⭐⭐⭐ EDIT THESE TWO LINES ⭐⭐⭐
NUM_TRAIN_SAMPLES = 100  # ← CHANGE THIS
NUM_TEST_SAMPLES = 20    # ← CHANGE THIS
# ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐

total_needed = NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES
if total_needed > len(dataset):
    print(f"⚠️  Adjusting to 90/10 split...")
    NUM_TRAIN_SAMPLES = int(len(dataset) * 0.9)
    NUM_TEST_SAMPLES = len(dataset) - NUM_TRAIN_SAMPLES

train_indices = list(range(NUM_TRAIN_SAMPLES))
test_indices = list(range(NUM_TRAIN_SAMPLES, NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES))

train_dataset = dataset.select(train_indices)
eval_dataset = dataset.select(test_indices)

print(f"✓ Train: {len(train_dataset)} samples")
print(f"✓ Test: {len(eval_dataset)} samples\n")

# ════════════════════════════════════════════════════════════
# CELL 7: DATA PROCESSOR
# ════════════════════════════════════════════════════════════

class ThaiHandwritingDataProcessor:
    def __init__(self, processor, tokenizer, config):
        self.processor = processor
        self.tokenizer = tokenizer
        self.config = config

    def process_example(self, example):
        try:
            image = example['image']
            text = example['text']

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Recognize the Thai text in this image."}
                ]
            }]

            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            response_tokens = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.config.MAX_LENGTH
            )

            input_ids = inputs['input_ids'][0].tolist()
            response_input_ids = response_tokens['input_ids']

            full_input_ids = input_ids + response_input_ids + [self.tokenizer.pad_token_id]
            labels = [-100] * len(input_ids) + response_input_ids + [self.tokenizer.pad_token_id]

            if len(full_input_ids) > self.config.MAX_LENGTH:
                full_input_ids = full_input_ids[:self.config.MAX_LENGTH]
                labels = labels[:self.config.MAX_LENGTH]

            attention_mask = [1] * len(full_input_ids)
            pad_length = self.config.MAX_LENGTH - len(full_input_ids)
            full_input_ids += [self.tokenizer.pad_token_id] * pad_length
            labels += [-100] * pad_length
            attention_mask += [0] * pad_length

            return {
                'input_ids': torch.tensor(full_input_ids),
                'attention_mask': torch.tensor(attention_mask),
                'labels': torch.tensor(labels),
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'image_grid_thw': inputs['image_grid_thw'].squeeze(0),
            }
        except:
            return None

data_processor = ThaiHandwritingDataProcessor(processor, tokenizer, config)
print("✓ Data processor ready\n")

# ════════════════════════════════════════════════════════════
# CELL 8: PROCESS DATASETS
# ════════════════════════════════════════════════════════════

print("Processing datasets...\n")

train_dataset_proc = train_dataset.map(
    data_processor.process_example,
    remove_columns=['image', 'text', 'label_file'],
    num_proc=2,
)

eval_dataset_proc = eval_dataset.map(
    data_processor.process_example,
    remove_columns=['image', 'text', 'label_file'],
    num_proc=2,
)

train_dataset_proc = train_dataset_proc.filter(lambda x: x is not None)
eval_dataset_proc = eval_dataset_proc.filter(lambda x: x is not None)

print(f"✓ Processed train: {len(train_dataset_proc)}")
print(f"✓ Processed eval: {len(eval_dataset_proc)}\n")

# ════════════════════════════════════════════════════════════
# CELL 9: TRAINING SETUP
# ════════════════════════════════════════════════════════════

print("Setting up training...\n")

training_args = TrainingArguments(
    output_dir=config.OUTPUT_DIR,
    per_device_train_batch_size=config.BATCH_SIZE,
    per_device_eval_batch_size=config.BATCH_SIZE,
    gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
    learning_rate=config.LEARNING_RATE,
    num_train_epochs=config.NUM_EPOCHS,
    warmup_steps=config.WARMUP_STEPS,
    logging_steps=10,
    save_steps=50,
    eval_steps=50,
    save_strategy="steps",
    evaluation_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="none",
    bf16=True,
    gradient_checkpointing=True,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
)

print("✓ Training setup complete\n")

# ════════════════════════════════════════════════════════════
# CELL 10: TRAIN!
# ════════════════════════════════════════════════════════════

print("="*60)
print("STARTING TRAINING")
print("="*60)
print()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_proc,
    eval_dataset=eval_dataset_proc,
    data_collator=data_collator,
)

train_result = trainer.train()

print()
print("="*60)
print("✅ TRAINING COMPLETED!")
print("="*60)
print(f"Final loss: {train_result.training_loss:.4f}\n")

# ════════════════════════════════════════════════════════════
# CELL 11: SAVE MODEL
# ════════════════════════════════════════════════════════════

print("Saving model...\n")

model.save_pretrained(f"{config.OUTPUT_DIR}/final_model")
processor.save_pretrained(f"{config.OUTPUT_DIR}/processor")
tokenizer.save_pretrained(f"{config.OUTPUT_DIR}/tokenizer")

print(f"✓ Model saved to: {config.OUTPUT_DIR}")
print(f"  - LoRA: {config.OUTPUT_DIR}/final_model")
print(f"  - Processor: {config.OUTPUT_DIR}/processor")
print(f"  - Tokenizer: {config.OUTPUT_DIR}/tokenizer\n")

print("🎉 Fine-tuning complete!\n")