import os
from datasets import Dataset
from datasets import load_dataset
from progressbar import progressbar
import numpy as np
from pycocotools import mask as mask_utils
from skimage import measure
import cv2
import json
import sys

# Add project root to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.parser import get_common_parser


def mask_to_coco_annotations(mask, image_id=0, image_index=None, starting_ann_id=0):
    annotations = []
    label_ids = np.unique(mask)
    ann_id = starting_ann_id

    image_dict = {
        'image': {
            'image_id': image_id,
             'license': 1,
             'file_name': f'{image_index}.jpg',
             'height': 512,
             'width': 512
        },
        'annotations': []
    }

    for label_id in label_ids:
        if label_id == 0:
            continue  # assuming 0 is background

        binary_mask = (mask == label_id).astype(np.uint8)
        if binary_mask.sum() == 0:
            continue

        # Encode in RLE
        rle = mask_utils.encode(np.asfortranarray(binary_mask))
        area = mask_utils.area(rle).item()
        bbox = mask_utils.toBbox(rle).tolist()

        annotation = {
            "id": ann_id,
            # "image_id": image_id,
            "category_id": int(label_id),
            "segmentation": {
                "size": list(binary_mask.shape),
                "counts": rle["counts"].decode("utf-8"),
            },
            "area": area,
            "bbox": bbox,
            "iscrowd": 0,
        }
        annotations.append(annotation)
        ann_id += 1
    image_dict['annotations'] = annotations
    return image_dict


def create_folder_structure(dataset_name, destination_folder=None):
    if destination_folder is None:
        destination_folder = './SAM_ResNet'
    dataset_folder = os.path.join(destination_folder, 'datasets', dataset_name.split("/")[-1].lower())
    dataset_folder += '-seg-sam'
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    if not os.path.exists(f"{dataset_folder}/train"):
        os.makedirs(f"{dataset_folder}/train")
    if not os.path.exists(f"{dataset_folder}/test"):
        os.makedirs(f"{dataset_folder}/test")
    if not os.path.exists(f"{dataset_folder}/valid"):
        os.makedirs(f"{dataset_folder}/valid")
    return dataset_folder


def fill_folder_structure(folder, dataset=None):
    for key in dataset.keys():
        counter = 0
        for row in progressbar(dataset[key]):
            mask = np.array(row['annotation'])
            image_id = row['page_id']
            image = row['image']
            if key == 'train':
                folder_dir = f'{folder}/train'
            elif key == 'test' and counter == 0:
                folder_dir = f'{folder}/test'
            elif key == 'test' and counter > 0:
                folder_dir = f'{folder}/valid'
            else:
                raise ValueError('Invalid key')
            if not os.path.exists(folder_dir):
                os.makedirs(folder_dir)
            image.save(f"{folder_dir}/{counter}.jpg")
            annotations = mask_to_coco_annotations(mask, image_id=image_id, image_index=counter)
            with open(f"{folder_dir}/{counter}.json", 'w') as f:
                json.dump(annotations, f)
            counter += 1







if __name__ == '__main__':
    parser = get_common_parser('Common parser')
    args = parser.parse_args()
    classification = args.classification
    dataset_name = f"gerbejon/WebClasSeg25-visual-{classification}"
    dataset = load_dataset(dataset_name)

    # folder = create_folder_structure(dataset_name=dataset_name, destination_folder='/home/ubuntu/sam_2')
    folder = create_folder_structure(dataset_name=dataset_name)

    # row = np.array(dataset['train'][0])
    # mask = row['annotation']
    # res = mask_to_coco_annotations(mask)
    fill_folder_structure(folder, dataset)