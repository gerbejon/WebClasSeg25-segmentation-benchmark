# train_roberta.py
# Example: python train_roberta.py --classification mc --modelversion v1

import os
import sys
import json
import torch
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# Add project root to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import login
from tools.parser import get_common_parser  # Reuse your shared parser

# -----------------------------
# Argument Parsing
# -----------------------------
# args = parse_args("Train RoBERTa classifier for WebClasSeg25 dataset")
parser = get_common_parser("Train RoBERTa classifier for WebClasSeg25 dataset")
args = parser.parse_args()

classification = args.classification
modelversion = args.modelversion
resume = args.resume
epochs = args.epochs if args.epochs is not None else 4
batch_size = args.batch_size if args.batch_size is not None else 8
lr = args.lr if args.lr is not None else 2e-5
output_dir = args.output_dir if args.output_dir is not None else f"./roBERTa/results_{classification}_{modelversion}"
push_to_hub = args.push_to_hub

# -----------------------------
# Dataset Loading
# -----------------------------
ds_name = f'gerbejon/WebClasSeg25-html-nodes-{classification}-balanced'
print(f"Starting training with {ds_name} dataset")

dataset = load_dataset(ds_name)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Number of GPUs:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# -----------------------------
# Hugging Face Authentication
# -----------------------------
with open('./credentials.json', 'r') as f:
    token_dict = json.load(f)

hugginface_username = token_dict['huggingface']['username']
login(token_dict['huggingface']['write'])

# -----------------------------
# Tokenizer
# -----------------------------
model_dir = f"roberta-html-nodes-{classification}-classifier-{modelversion}"
if os.path.exists(os.path.join(model_dir, 'tokenizer.json')):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
else:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# -----------------------------
# Tokenization
# -----------------------------
def tokenize_function(example):
    if modelversion == 'v1':
        return tokenizer(example['path_translated_class'], padding='max_length', truncation=True, max_length=512)
    elif modelversion == 'v2':
        return tokenizer(example['path_class'], padding='max_length', truncation=True, max_length=512)
    else:
        return tokenizer(example['path_translated_class'], padding='max_length', truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# -----------------------------
# Label Encoding
# -----------------------------
if isinstance(dataset["train"].features["y"], ClassLabel):
    num_labels = dataset["train"].features["y"].num_classes
    label2id = {label: i for i, label in enumerate(dataset["train"].features["y"].names)}
    id2label = {v: k for k, v in label2id.items()}
else:
    labels = sorted(set(dataset["train"]["y"]))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    def encode_labels(example):
        example["label"] = label2id[example["y"]]
        return example

    tokenized_datasets = tokenized_datasets.map(encode_labels)
    num_labels = len(label2id)

tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# -----------------------------
# Model Initialization
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# Training Setup
# -----------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    push_to_hub=push_to_hub,
    hub_model_id=f"{hugginface_username}/roberta-html-nodes-{classification}-classifier-{modelversion}",
    hub_strategy="every_save",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

# -----------------------------
# Training
# -----------------------------
if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0 and resume:
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)

# -----------------------------
# Save & Push Model
# -----------------------------
# model_dir = f"roberta-html-nodes-{classification}-classifier-{modelversion}"
trainer.save_model(model_dir)
if push_to_hub:
    trainer.push_to_hub(f"{hugginface_username}/{model_dir}")
