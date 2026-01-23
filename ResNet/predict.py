"""
Prediction / Inference script for ResNet / SegFormer model (WebClasSeg25)
Structure mimics the LongCoder textual inference script.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
import pandas as pd
from progressbar import progressbar
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from accelerate.test_utils.testing import get_backend
# Add project root to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.parser import get_common_parser
from tools.img_plots import blend_segmentation_mask
from tools.utils import get_confusion_matrix, mask_to_polygon
from torchvision.transforms import ColorJitter
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------
def load_labels(classification: str):
    """Load label maps."""
    label_dir = Path("./data")
    with open(label_dir / f"label2id_{classification}.json") as f:
        label2id = json.load(f)

    id2label_path = label_dir / f"id2label_{classification}.json"
    if id2label_path.exists():
        with open(id2label_path) as f:
            id2label = json.load(f)
    else:
        id2label = {int(v): k for k, v in label2id.items()}
        with open(id2label_path, "w") as f:
            json.dump(id2label, f)

    return label2id, id2label


def setup_model_and_processor(model_dir: str, num_labels: int, id2label, label2id):
    """Load segmentation model and processor."""
    processor = AutoImageProcessor.from_pretrained(model_dir, num_labels=num_labels)
    model = AutoModelForSemanticSegmentation.from_pretrained(
        model_dir, id2label=id2label, label2id=label2id
    )
    device, _, _ = get_backend()
    model.to(device)
    model.eval()
    return processor, model, device

# ---------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------
def preprocess_image(image, processor, jitter=None):
    """Apply optional jitter and prepare tensor for model."""
    if jitter is not None:
        image = jitter(image)
    return processor(image, return_tensors="pt")


def predict_image(image, processor, model, device):
    """Run model inference and return predicted segmentation mask."""
    encoding = preprocess_image(image, processor)
    pixel_values = encoding.pixel_values.to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits.cpu()
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    return pred_seg

# ---------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------
def run_prediction(args):
    classification = args.classification
    modelversion = args.modelversion
    resume = args.resume
    epochs = args.epochs if args.epochs is not None else 80
    batch_size = args.batch_size if args.batch_size is not None else 2
    lr = args.lr if args.lr is not None else 6e-5
    model_out = f"segformer-websegcl25-2000-{classification}-{modelversion}"
    model_dir = args.output_dir if args.output_dir is not None else f"./ResNet/{model_out}"
    model_dir = model_dir if os.path.exists(model_dir) else f"gerbejon/segformer-websegcl25-2000-{classification}-{modelversion}"
    push_to_hub = args.push_to_hub
    # Setup output directory
    rootdir = os.getcwd()
    outdir = os.path.join(rootdir, f"ResNet/results/{args.modelversion}/result_out")
    os.makedirs(outdir, exist_ok=True)

    # Load label maps
    label2id, id2label = load_labels(args.classification)
    num_labels = len(id2label)

    # Load model and processor
    processor, model, device = setup_model_and_processor(model_dir, num_labels, id2label, label2id)
    print(f"Using model: {model_dir}")

    # Optional: define ColorJitter for inference if needed
    jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

    # Load dataset
    dataset = load_dataset(f"gerbejon/WebClasSeg25-visual-{classification}")
    test_ds = dataset["test"]
    print(f"Loaded dataset: gerbejon/WebClasSeg25-visual-{classification}")

    # Prediction storage
    results_page = {}
    cm_df = None
    all_results = []

    for row in progressbar(test_ds):
        page_id = row["page_id"]
        image = row["image"]
        annotation = np.array(row["annotation"])

        # Predict segmentation
        pred_seg = predict_image(image, processor, model, device)



        # # Blend and save segmentation masks
        # blend_segmentation_mask(
        #     img_mask=annotation + 1,
        #     segm=args.classification,
        #     img=image,
        #     outfile=os.path.join(outdir, f"{page_id}_seg.png"),
        # )

        # Save ground truth and prediction CSVs
        j = 0
        while True:
            annot_file = os.path.join(outdir, f"y_{page_id}_annotator{j}.csv")
            pred_file = os.path.join(outdir, f"y_hat_{page_id}_annotator{j}.csv")
            if os.path.exists(annot_file):
                j += 1
            else:
                pd.DataFrame(annotation).to_csv(annot_file, index=False)
                pd.DataFrame(pred_seg).to_csv(pred_file, index=False)
                break

        # Compute confusion matrix
        y_label = [id2label[str(int(i))] for i in annotation.flatten()]
        y_hat_label = [id2label[str(int(i))] for i in pred_seg.flatten()]
        tmp_df = get_confusion_matrix(y_true=y_label, y_pred=y_hat_label)

        if cm_df is None:
            cm_df = tmp_df.copy()
        else:
            cm_df.index = cm_df.index.astype(str)
            tmp_df.index = tmp_df.index.astype(str)
            cm_df = cm_df.add(tmp_df, fill_value=0)

        # Compute F1 score
        f1_vals = f1_score(annotation.flatten(), pred_seg.flatten(), average=None)
        sorted_ids = sorted(set(annotation.flatten()))
        tmp_dict = {id2label[str(int(sorted_ids[idx]))]: f1_vals[idx] for idx in range(len(sorted_ids))}
        all_results.append(tmp_dict)

        # Polygon conversion for JSON output
        img_dict = {
            "id": page_id,
            "width": image.size[0],
            "height": image.size[1],
            "segmentations": {"predicted": []},
            "nodes": {"predicted": []},
        }
        for cl in np.unique(annotation):
            xy_polygon = mask_to_polygon(mask=annotation, tolerance=0.5)
            img_dict["segmentations"]["predicted"].append({
                "polygon": [[xy_polygon]],
                "tagType": id2label[str(int(cl))],
            })
        results_page[page_id] = img_dict

    # Save F1 results
    f1_df = pd.DataFrame(all_results, index=[row["page_id"] for row in test_ds])
    f1_df.to_csv(os.path.join(outdir, f"f1_results_{args.classification}_{args.modelversion}.csv"), index=True)

    # Save summary
    summary_df = pd.concat([
        f1_df.apply("mean", axis=0),
        f1_df.apply("std", axis=0)
    ], axis=1)
    summary_df.to_csv(os.path.join(outdir, f"f1_summary_{args.classification}_{args.modelversion}.csv"))

    print(f"Done! Predictions saved in {outdir}")

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = get_common_parser("Run inference using fine-tuned ResNet / SegFormer")
    args = parser.parse_args()
    run_prediction(args)
