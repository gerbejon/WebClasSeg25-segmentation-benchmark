"""
Semantic segmentation training script for WebClasSeg25
Based on SegFormer fine-tuning:
https://huggingface.co/docs/transformers/tasks/semantic_segmentation
"""

# =========================
# Imports
# =========================
import os
import sys
import json
from pathlib import Path
# from socket import gethostname

import numpy as np
import torch
from torch import nn
from PIL import Image

from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    TrainingArguments,
    Trainer,
)
from torchvision.transforms import ColorJitter
import evaluate
from huggingface_hub import upload_folder
# from accelerate.test_utils.testing import get_backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.parser import get_common_parser  # Reuse your shared parser
# from utils.div import ade_palette


# -----------------------------
# Argument Parsing
# -----------------------------
# args = parse_args("Train RoBERTa classifier for WebClasSeg25 dataset")
parser = get_common_parser("Train RoBERTa classifier for WebClasSeg25 dataset")
args = parser.parse_args()

classification = args.classification
modelversion = args.modelversion
resume = args.resume
epochs = args.epochs if args.epochs is not None else 80
batch_size = args.batch_size if args.batch_size is not None else 2
lr = args.lr if args.lr is not None else 6e-5
model_out = f"segformer-websegcl25-2000-{classification}-{modelversion}"
output_dir = args.output_dir if args.output_dir is not None else f"./ResNet/{model_out}"
push_to_hub = args.push_to_hub




# =========================
# Configuration
# =========================
# TASK_DESCR = {
#     "mc": "training maturity segmentation",
#     "fc": "training functional segmentation",
# }

CHECKPOINT = "nvidia/mit-b0"
# modelversion = "v3"
LABEL_DIR = Path("./data")

# DEFAULT_TASK = "mc"
# LOCAL_HOSTNAME = "gerj-Precision-5560"


# # =========================
# # Task selection
# # =========================
# def get_task_from_args() -> str:
#     if len(sys.argv) > 1:
#         task = sys.argv[1].lower()
#         if task in TASK_DESCR:
#             print(TASK_DESCR[task])
#             return task
#         raise ValueError("Invalid task provided (use 'fc' or 'mc')")
#     return DEFAULT_TASK if gethostname() == LOCAL_HOSTNAME else None


# classification = get_task_from_args()
if classification is None:
    raise RuntimeError("No classification task provided")



# =========================
# Dataset transforms
# =========================
def train_transforms(batch):
    images = [jitter(img) for img in batch["image"]]
    labels = batch["annotation"]
    return image_processor(images, labels)


def val_transforms(batch):
    return image_processor(batch["image"], batch["annotation"])


# =========================
# Metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    with torch.no_grad():
        logits = torch.from_numpy(logits)
        logits = nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        preds = logits.cpu().numpy()

        metrics = metric.compute(
            predictions=preds,
            references=labels,
            num_labels=num_labels,
            ignore_index=255,
            reduce_labels=False,
        )

    return {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in metrics.items()
    }


# =========================
# Dataset loading
# =========================
ds = load_dataset(
    f"gerbejon/WebClasSeg25-visual-{classification}",
    download_mode="force_redownload",
)

train_ds = ds["train"]
test_ds = ds["test"]

# # Quick visual sanity check
# Image.blend(
#     train_ds[0]["image"].convert("RGBA"),
#     train_ds[0]["annotation"].convert("RGBA"),
#     alpha=0.7,
# ).show()


# =========================
# Labels
# =========================
with open(LABEL_DIR / f"label2id_{classification}.json") as f:
    label2id = json.load(f)

id2label_path = LABEL_DIR / f"id2label_{classification}.json"
if id2label_path.exists():
    with open(id2label_path) as f:
        id2label = json.load(f)
else:
    id2label = {int(v): k for k, v in label2id.items()}
    with open(id2label_path, "w") as f:
        json.dump(id2label, f)

num_labels = len(id2label)


# =========================
# Preprocessing
# =========================
# if os.path.exists(output_dir) and args.resume:
# #     image_processor = AutoImageProcessor.from_pretrained(
# #         output_dir,
# #         num_labels=num_labels,
# #     )
# # else:
image_processor = AutoImageProcessor.from_pretrained(
    CHECKPOINT,
    num_labels=num_labels,
)
image_processor.save_pretrained(output_dir)

jitter = ColorJitter(
    brightness=0.25,
    contrast=0.25,
    saturation=0.25,
    hue=0.1,
)

train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)


# =========================
# Evaluation
# =========================
metric = evaluate.load("mean_iou")


# =========================
# Training
# =========================
model = AutoModelForSemanticSegmentation.from_pretrained(
    CHECKPOINT,
    id2label=id2label,
    label2id=label2id,
)
model.save_pretrained(output_dir)

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    save_total_limit=3,
    logging_steps=1,
    logging_dir="./logs",
    eval_accumulation_steps=5,
    remove_unused_columns=False,
    push_to_hub=push_to_hub,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

trainer.train(
    resume_from_checkpoint=args.resume,
)

if push_to_hub:
    trainer.push_to_hub()

    upload_folder(
        folder_path=output_dir,
        repo_id=f"gerbejon/{model_out}",
        path_in_repo=".",
    )

#
# # =========================
# # Local inference (optional)
# # =========================
# if gethostname() == LOCAL_HOSTNAME:
#     image = ds["test"][0]["image"]
#
#     device, _, _ = get_backend()
#     model.to(device)
#
#     encoding = image_processor(image, return_tensors="pt")
#     pixel_values = encoding.pixel_values.to(device)
#
#     with torch.no_grad():
#         logits = model(pixel_values=pixel_values).logits
#
#     logits = nn.functional.interpolate(
#         logits.cpu(),
#         size=image.size[::-1],
#         mode="bilinear",
#         align_corners=False,
#     )
#
#     pred = logits.argmax(dim=1)[0].numpy()
#
#     palette = np.array(ade_palette())
#     color_seg = palette[pred]
#
#     overlay = (0.5 * np.array(image) + 0.5 * color_seg).astype(np.uint8)
#
#     plt.figure(figsize=(15, 10))
#     plt.imshow(overlay)
#     plt.axis("off")
#     plt.show()
