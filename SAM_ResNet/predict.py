"""
Prediction / Inference script for SAM + ResNet18 pipeline (WebClasSeg25)
Structure intentionally mirrors SegFormer / ResNet inference scripts.
"""

import os
import sys
import json
import copy
import shutil
import re
import numpy as np
import pandas as pd
from PIL import Image
from progressbar import progressbar
from datasets import load_dataset

# Project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.parser import get_common_parser
from tools._f1_score import f1_score_mask
from visual_tools import mask_to_polygon
from SAM.predict import SAM_segmentation
from ResNet18.predict import ResNet18_segment_classifier


# ---------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------
def load_labels(classification: str):
    with open(f"../id2label_{classification}.json", "r") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    return id2label


def setup_models(classification: str):
    resnet_model = ResNet18_segment_classifier(segm=classification)
    sam_model = SAM_segmentation(segm=classification)
    return sam_model, resnet_model


# ---------------------------------------------------------------------
# Prediction pipeline
# ---------------------------------------------------------------------
def run_prediction(args):
    classification = args.classification
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    save_results = args.save_results

    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting SAM + ResNet inference for segmentation={classification}")

    # Load dataset + labels
    ds = load_dataset(f"gerbejon/WebClasSeg25-visual-{classification}")
    id2label = load_labels(classification)

    # Load models
    sam_model, resnet_model = setup_models(classification)

    # Collect test images
    test_imgs = [
        f for f in os.listdir(dataset_dir)
        if f.endswith(".jpg")
    ]

    result_dict = {}
    f1_list = []
    img_list = []

    for img_name in progressbar(test_imgs):
        img_id = img_name.split(".")[0]
        image_path = os.path.join(dataset_dir, img_name)
        image = Image.open(image_path).convert("RGB")

        img_dict = {
            "id": img_id,
            "width": image.size[0],
            "height": image.size[1],
            "segmentations": {"predicted": []},
            "nodes": {"predicted": []},
        }

        # -----------------------------------------------------------------
        # SAM → ResNet inference
        # -----------------------------------------------------------------
        mask = np.zeros((image.size[1], image.size[0]), dtype=np.int32)
        sam_result = sam_model.predict(image=np.array(image))

        for segment in sam_result:
            binary_mask = segment["segmentation"]
            masked_image = binary_mask[:, :, None] * np.array(image)

            cls_id = resnet_model.predict(
                image=Image.fromarray(masked_image)
            )

            mask[binary_mask] = cls_id + 1

            polygon = mask_to_polygon(binary_mask, tolerance=0.5)
            img_dict["segmentations"]["predicted"].append({
                "polygon": [[polygon]],
                "tagType": id2label[cls_id + 1],
            })

        result_dict[img_id] = img_dict

        # -----------------------------------------------------------------
        # F1 computation
        # -----------------------------------------------------------------
        for row in ds["test"]:
            if row["page_id"] == img_id:
                mask_true = np.array(row["annotation"])
                break

        f1_dicts = f1_score_mask(
            mask=mask,
            segm=classification,
            img_id=img_id,
            ds=ds
        )

        for d in f1_dicts:
            f1_list.append(d)
            img_list.append(img_id)

    # -----------------------------------------------------------------
    # Save F1 scores
    # -----------------------------------------------------------------
    df = pd.concat([pd.Series(row) for row in f1_list], axis=1).T
    df.columns = [id2label[int(c)] for c in df.columns]
    df.index = img_list

    f1_dir = os.path.join(output_dir, "f1_scores")
    os.makedirs(f1_dir, exist_ok=True)
    df.to_csv(os.path.join(f1_dir, f"f1_sam_resnet_{classification}.csv"))

    # -----------------------------------------------------------------
    # Save JSON predictions
    # -----------------------------------------------------------------
    if save_results:
        with open(os.path.join(output_dir, "predictions.json"), "w") as f:
            json.dump(result_dict, f, indent=2)

    print(f"Done! Results saved to {output_dir}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = get_common_parser(
        "Run inference using SAM + ResNet18 segmentation"
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing test images",
    )

    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save predicted polygons as JSON",
    )

    args = parser.parse_args()
    run_prediction(args)
