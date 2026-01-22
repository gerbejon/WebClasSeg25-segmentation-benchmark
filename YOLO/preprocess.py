import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.parser import get_yolo_parser
from datasets import Dataset
from datasets import load_dataset
import numpy as np
import cv2
from progressbar import progressbar

def masks_to_yolo_segments(mask, image_width, image_height):
    segments = []
    class_ids = np.unique(mask)
    if segm == 'fc':
        class_ids = class_ids[class_ids != 0]  # exclude background
    else:
        class_ids = class_ids[class_ids != 8]

    for class_id in class_ids:
        binary_mask = (mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if len(cnt) < 3:  # need at least 3 points for a polygon
                continue

            segment = [str(class_id - 1)] if segm == 'fc' else [str(class_id)] # YOLO class indices usually start at 0
            for point in cnt.squeeze():
                x = point[0] / image_width
                y = point[1] / image_height
                segment.extend([f"{x:.6f}", f"{y:.6f}"])

            segments.append(" ".join(segment))

    return segments

def masks_to_yolo_labels(mask, image_width, image_height):
    labels = []
    class_ids = np.unique(mask)
    class_ids = class_ids[class_ids != 0]  # exclude background

    for class_id in class_ids:
        binary_mask = (mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Normalize
            x_center = (x + w / 2) / image_width
            y_center = (y + h / 2) / image_height
            w /= image_width
            h /= image_height
            labels.append(f"{class_id - 1} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    return labels

def create_folder_structure(dataset_name):
    dataset_folder = os.path.join('./YOLO/datasets', dataset_name.split("/")[-1].lower())
    dataset_folder += '-seg'
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    if not os.path.exists(f"{dataset_folder}/images/train"):
        os.makedirs(f"{dataset_folder}/images/train")
    if not os.path.exists(f"{dataset_folder}/images/test"):
        os.makedirs(f"{dataset_folder}/images/test")
    if not os.path.exists(f"{dataset_folder}/labels/train"):
        os.makedirs(f"{dataset_folder}/labels/train")
    if not os.path.exists(f"{dataset_folder}/labels/test"):
        os.makedirs(f"{dataset_folder}/labels/test")
    return dataset_folder


def preprocess(dataset_name: str):
    folder = create_folder_structure(dataset_name)
    dataset = load_dataset(dataset_name)

    for location in dataset.keys():
        for row in progressbar(dataset[location]):
            page_id = row["page_id"]
            image = row["image"]
            img_count = len([img_name for img_name in os.listdir(f"{folder}/images/{location}/") if bool(re.match(f'^{page_id}.+', img_name))])
            image.save(f"{folder}/images/{location}/{page_id}_{img_count}.jpg")
            mask = np.array(row["annotation"])
            if segmentation:
                yoloy_segments = masks_to_yolo_segments(mask, mask.shape[0], mask.shape[1])
                with open(f"{folder}/labels/{location}/{page_id}_{img_count}.txt", 'w') as f:
                    for line in yoloy_segments:
                        f.write(f"{line}\n")
            else:
                yoloy_labels = masks_to_yolo_labels(mask, mask.shape[0], mask.shape[1])
                with open(f"{folder}/labels/{location}/{page_id}_{img_count}.txt", 'w') as f:
                    for line in yoloy_labels:
                        f.write(f"{line}\n")

def create_txt_files(dataset_name: str):
    folder = create_folder_structure(dataset_name)
    for subfolder in  os.listdir(os.path.join(folder, "images")):
        with open(f"{folder}/{subfolder}.txt", 'w') as f:
            for file in os.listdir(os.path.join(folder, "images", subfolder)):
                f.write(f'./images/{subfolder}/{file}\n')








if __name__ == '__main__':
    parser = get_yolo_parser('Parser for YOLO model')
    args = parser.parse_args()
    print(args)
    classification = args.classification
    segmentation = True
    # dataset_name = "gerbejon/WebClasSeg25-visual-fc"
    dataset_name = f"gerbejon/WebClasSeg25-visual-{classification}"
    segm = dataset_name.split("-")[-1]


    dataset = load_dataset(dataset_name)
    create_folder_structure(dataset_name=dataset_name)
    # mask = np.array(dataset["train"][0]['annotation'])
    # res = masks_to_yolo_labels(mask, mask.shape[0], mask.shape[1])
    preprocess(dataset_name=dataset_name)
    create_txt_files(dataset_name=dataset_name)