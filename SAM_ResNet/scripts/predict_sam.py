import torch
import os
# os.getcwd()
# os.chdir("/home/ubuntu/sam_2/sam2")
import sys
sys.path.append(os.path.abspath("/home/ubuntu/sam_2/sam2"))
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import supervision as sv
# import os
import random
from PIL import Image
import numpy as np

class SAM_segmentation:
    def __init__(self, segm, checkpoint=None, model_cfg=None, device="cuda"):
        self.segm = segm
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if checkpoint is None:
            checkpoint = f"/home/ubuntu/dataset_benchmark/SAM/sam2/sam2_logs/configs/sam2.1-{self.segm}/train.yaml/checkpoints/checkpoint.pt"
        # model_cfg = "/home/ubuntu/sam_2/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
        if model_cfg is None:
            model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        sam2 = build_sam2(model_cfg, checkpoint, device=device)
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2)

    def predict(self, image =None, image_dir=None):
        # validation
        # validation_set = os.listdir("/home/ubuntu/sam_2/datasets_sam/webclasseg25-visual-{segm}-seg/valid")

        # choose random with .json extension
        # image = random.choice([img for img in validation_set if img.endswith(".jpg")])
        # image = os.path.join("/home/ubuntu/sam_2/datasets_sam/webclasseg25-visual-{segm}-seg/valid", image)
        if image is None:
            if image_dir is None:
                image_dir = f'/home/ubuntu/sam_2/datasets_sam/webclasseg25-visual-{self.segm}-seg/valid/319.jpg'
            image = np.array(Image.open(image_dir).convert("RGB"))
        result = self.mask_generator.generate(image)
        return result

    def predict_many(self, image_dir=None):
        if image_dir is None:
            image_dir = f"/home/ubuntu/sam_2/datasets_sam/webclasseg25-visual-{self.segm}-seg/valid"
        validation_set = os.listdir(image_dir)

        # image = random.choice([img for img in validation_set if img.endswith(".jpg")])
        images = [os.path.join(f"/home/ubuntu/sam_2/datasets_sam/webclasseg25-visual-{self.segm}-seg/valid", image) for image in validation_set if image.endswith(".jpg")]
        for image in images:
            yield self.predict(image)


if __name__ == '__main__':
    segm = 'mc'
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint = f"sam2/sam2_logs/configs/sam2.1-{segm}/train.yaml/checkpoints/checkpoint.pt"
    # model_cfg = "/home/ubuntu/sam_2/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam2 = build_sam2(model_cfg, checkpoint, device="cuda")
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    # validation
    validation_set = os.listdir(f"/home/ubuntu/sam_2/datasets_sam/webclasseg25-visual-{segm}-seg/valid")

    # choose random with .json extension
    image = random.choice([img for img in validation_set if img.endswith(".jpg")])
    image = os.path.join(f"/home/ubuntu/sam_2/datasets_sam/webclasseg25-visual-{segm}-seg/valid", image)
    opened_image = np.array(Image.open(image).convert("RGB"))
    result = mask_generator.generate(opened_image)
