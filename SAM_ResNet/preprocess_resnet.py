import json
import os

import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils
from progressbar import progressbar
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.parser import get_common_parser

def create_folder_structure(dataset_name, destination_folder=None):
    if destination_folder is None:
        destination_folder = './SAM_ResNet'
    dataset_folder = os.path.join(destination_folder, 'datasets', dataset_name.split("/")[-1].lower())
    dataset_folder += '-seg-resnet'
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    if not os.path.exists(f"{dataset_folder}/train"):
        os.makedirs(f"{dataset_folder}/train")
    if not os.path.exists(f"{dataset_folder}/test"):
        os.makedirs(f"{dataset_folder}/test")
    if not os.path.exists(f"{dataset_folder}/valid"):
        os.makedirs(f"{dataset_folder}/valid")
    return dataset_folder

def create_dataset(sam_dataset_dir, dataset_dir):
    counter = 0
    for folder in os.listdir(sam_dataset_dir):
        metadata = []
        for file in progressbar(os.listdir(os.path.join(sam_dataset_dir, folder))):
            if file.endswith(".jpg"):
                image = Image.open(os.path.join(sam_dataset_dir, folder, file))
                with open(os.path.join(sam_dataset_dir, folder, file.replace('.jpg', '.json')), 'r') as f:
                    seg_dict = json.load(f)

                for segment in seg_dict['annotations']:
                    metadata.append({
                        'file_name': f'{str(counter)}.png',
                        'image_id': seg_dict['image']['image_id'],
                        'label': segment['category_id']
                    })

                    category_id = segment['category_id']
                    mask = mask_utils.decode(segment['segmentation'])
                    masked_image = mask[:,:,None] * image
                    if not str(category_id) in os.listdir(os.path.join(dataset_dir, folder)):
                        os.mkdir(os.path.join(dataset_dir, folder, str(category_id)))
                    # Image.fromarray(masked_image).save(os.path.join(dataset_dir, folder,  f'{str(counter)}.png'))
                    Image.fromarray(masked_image).save(os.path.join(dataset_dir, folder, str(category_id), f'{str(counter)}.png'))
                    counter += 1
        # pd.DataFrame(metadata).to_csv(os.path.join(dataset_dir, folder, 'metadata.csv'), index=False)

if __name__ == '__main__':
    parser = get_common_parser('Common parser')
    args = parser.parse_args()
    classification = args.classification
    dataset_name = f"gerbejon/WebClasSeg25-visual-{classification}"
    folder = create_folder_structure(dataset_name=dataset_name)
    sam_dataset_dir = folder.replace('resnet', 'sam')
    if not os.path.exists(sam_dataset_dir):
        raise f'{sam_dataset_dir} does not exist, please run preprocess_sam.py first'
    create_dataset(sam_dataset_dir=sam_dataset_dir, dataset_dir=folder)