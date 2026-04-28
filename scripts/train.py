"""
Reference: train.py — Fine-tune Gemma 2 2B for Banking Intent Classification using Unsloth
This is a REFERENCE script. Study it and write your own version.
"""

# ============================================================
# 1. Imports
# ============================================================
import os
import yaml
import json
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ============================================================
# 2. Load configuration from YAML
# ============================================================
# __file__ is the path to this script (scripts/train.py)
# We go up one level to reach the project root (banking-intent-unsloth/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config_path = os.path.join(PROJECT_ROOT, "configs", "train.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Extract config values into readable variables
MODEL_NAME      = config["model"]["name"]           # e.g. "unsloth/gemma-2-2b-it-bnb-4bit"
MAX_SEQ_LENGTH  = config["model"]["max_seq_length"]  # e.g. 256
LORA_R          = config["lora"]["r"]                # e.g. 16
LORA_ALPHA      = config["lora"]["alpha"]            # e.g. 32
LORA_DROPOUT    = config["lora"]["dropout"]          # e.g. 0.05

BATCH_SIZE      = config["training"]["batch_size"]             # e.g. 8
GRAD_ACCUM      = config["training"]["gradient_accumulation"]  # e.g. 4
EPOCHS          = config["training"]["epochs"]                 # e.g. 3
LEARNING_RATE   = config["training"]["learning_rate"]          # e.g. 2e-4
OPTIMIZER       = config["training"]["optimizer"]              # e.g. "adamw_8bit"
LR_SCHEDULER   = config["training"]["lr_scheduler"]           # e.g. "cosine"
WARMUP_RATIO    = config["training"]["warmup_ratio"]           # e.g. 0.1

OUTPUT_DIR      = os.path.join(PROJECT_ROOT, config["paths"]["output_dir"])
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, config["paths"]["train_data"])
VAL_DATA_PATH   = os.path.join(PROJECT_ROOT, config["paths"]["val_data"])
LABEL_MAP_PATH  = os.path.join(PROJECT_ROOT, config["paths"]["label_mapping"])

print(f"Config loaded from: {config_path}")
print(f"Model: {MODEL_NAME}")
print(f"Output: {OUTPUT_DIR}")

# ============================================================
# 3. Load model + tokenizer via Unsloth
# ============================================================
print("\n==== LOADING MODEL ====")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,       # Use 4-bit quantization (QLoRA)
    dtype=None,              # Auto-detect dtype
)

# IMPORTANT: Set pad_token if the tokenizer doesn't have one
# Gemma uses <eos> as pad by default, but just in case:
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded: {MODEL_NAME}")
print(f"Vocab size: {len(tokenizer)}")

# ============================================================
# 4. Apply LoRA adapters
# ============================================================
print("\n==== APPLYING LoRA ====")

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention layers
        "gate_proj", "up_proj", "down_proj",      # MLP layers
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",  # memory optimization
    random_state=42,
)

# Print trainable parameters
model.print_trainable_parameters()

# ============================================================
# 5. Prepare dataset — format into instruction prompts
# ============================================================
print("\n==== PREPARING DATASET ====")

# Load CSVs
df_train = pd.read_csv(TRAIN_DATA_PATH)
df_val   = pd.read_csv(VAL_DATA_PATH)

# --- Data validation: drop NaN & ensure string types ---
df_train.dropna(subset=["text", "label_name"], inplace=True)
df_val.dropna(subset=["text", "label_name"], inplace=True)
df_train["text"]       = df_train["text"].astype(str)
df_train["label_name"] = df_train["label_name"].astype(str)
df_val["text"]         = df_val["text"].astype(str)
df_val["label_name"]   = df_val["label_name"].astype(str)

# Load label mapping for reference
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_mapping = json.load(f)

print(f"Train samples: {len(df_train)}")
print(f"Val samples:   {len(df_val)}")
print(f"Num labels:    {label_mapping['num_labels']}")

# ----------------------------------------------------------
# Define the prompt template
# This is the MOST IMPORTANT part — inference MUST use the
# exact same template (without the response for generation)
# ----------------------------------------------------------
PROMPT_TEMPLATE = """### Instruction: Classify the banking intent of the following customer message. Respond with only the intent label.
### Input: {text}
### Response: {label}"""

def format_prompt(row):
    """Format a single row into the instruction prompt with EOS token."""
    return PROMPT_TEMPLATE.format(
        text=str(row["text"]),
        label=str(row["label_name"]),
    ) + tokenizer.eos_token   # <-- EOS token tells model when to STOP generating

# Apply the template to create the "formatted_text" column
df_train["formatted_text"] = df_train.apply(format_prompt, axis=1)
df_val["formatted_text"]   = df_val.apply(format_prompt, axis=1)

# Preview one example
print("\n--- Example prompt ---")
print(df_train["formatted_text"].iloc[0])
print("--- End example ---")

# Convert to HuggingFace Datasets using pre-formatted text
# IMPORTANT: ensure every element is a pure str — tokenizer will crash on NaN/None
train_texts = [str(t) for t in df_train["formatted_text"].tolist()]
val_texts   = [str(t) for t in df_val["formatted_text"].tolist()]

train_dataset = Dataset.from_dict({"formatted_text": train_texts})
val_dataset   = Dataset.from_dict({"formatted_text": val_texts})

print(f"Dataset columns: {train_dataset.column_names}")

# ============================================================
# 6. Set up trainer (SFTTrainer from trl)
# ============================================================
print("\n==== SETTING UP TRAINER ====")

# formatting_func: Unsloth calls this with a single example (dict).
# Must always return a list of strings.
def formatting_func(example):
    return [example["formatted_text"]]

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Batch & accumulation
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,

    # Epochs & steps
    num_train_epochs=EPOCHS,
    warmup_steps=10,             # warmup_ratio is deprecated, use warmup_steps instead

    # Optimizer & scheduler
    learning_rate=LEARNING_RATE,
    optim=OPTIMIZER,
    lr_scheduler_type=LR_SCHEDULER,

    # Precision
    fp16=True,    # Use mixed precision (T4 supports fp16)
    bf16=False,   # T4 does NOT support bf16

    # Logging & evaluation
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,

    # Misc
    seed=42,
    report_to="none",   # Set to "wandb" if you use Weights & Biases
)


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    formatting_func=formatting_func,   # ← replaces dataset_text_field
    args=training_args,
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,   # Set True to pack multiple short examples into one sequence
)


# ============================================================
# 7. Train!
# ============================================================
print("\n==== TRAINING ====")
trainer_stats = trainer.train()

# Print training summary
print("\n==== TRAINING COMPLETE ====")
print(f"Total steps:    {trainer_stats.global_step}")
print(f"Training loss:  {trainer_stats.training_loss:.4f}")
print(f"Train runtime:  {trainer_stats.metrics['train_runtime']:.1f}s")

# ============================================================
# 8. Save the fine-tuned LoRA adapter + tokenizer
# ============================================================
print("\n==== SAVING MODEL ====")

save_path = os.path.join(OUTPUT_DIR, "final_checkpoint")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# Also copy the label mapping into the checkpoint dir for convenience
import shutil
shutil.copy(LABEL_MAP_PATH, os.path.join(save_path, "label_mapping.json"))

print(f"Model saved to: {save_path}")
print("Done! You can now use inference.py with this checkpoint.")

