import os
import sys
import json
import torch
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from huggingface_hub import login
from tools.parser import parse_args

# -----------------------------
# Argument Parsing
# -----------------------------
args = parse_args("Train LongCoder classifier for WebClasSeg25 dataset")

# Access them like normal
classification = args.classification
modelversion = args.modelversion
resume = args.resume
epochs = args.epochs if args.epochs is not None else 3
batch_size = args.batch_size if args.batch_size is not None else 4
lr = args.lr if args.lr is not None else 2e-5
output_dir = args.output_dir if args.output_dir is not None else f"./codeBERT/results_{classification}_{modelversion}"
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
model_dir = f'longcoder-html-nodes-{classification}-classifier-{modelversion}'
if os.path.exists(os.path.join(model_dir, 'tokenizer.json')):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
else:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/longcoder-base")


# -----------------------------
# Tokenization
# -----------------------------
def tokenize_function(example):
    if modelversion == 'v1':
        return tokenizer(
            example["tag_head"] + ' ' + example['xpath'],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    else:
        # Placeholder for v2
        return {}


tokenized_datasets = dataset.map(tokenize_function, batched=False)


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


# -----------------------------
# Model Initialization
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/longcoder-base",
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
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    push_to_hub=push_to_hub,
    hub_model_id=f"{hugginface_username}/longcoder-html-nodes-{classification}-classifier-{modelversion}",
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
if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)


# -----------------------------
# Save & Push Model
# -----------------------------
trainer.save_model(model_dir)
trainer.push_to_hub(f'{hugginface_username}/{model_dir}')
