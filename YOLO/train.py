from ultralytics import YOLO
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.parser import parse_args

args = parse_args(model='yolo', description='Yolo training')
print(args)
classification = args.classification
modelversion = args.modelversion
output_dir = args.output_dir if args.output_dir is not None else f"./YOLO/models"
epochs = args.epochs if args.epochs is not None else 200
os.makedirs(output_dir, exist_ok=True)

model = YOLO("yolo11n.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data=f'./YOLO/webclasseg25-visual-{classification}-seg.yaml',
    epochs=epochs,  # Number of training epochs
    imgsz=512,  # Image size for training
    device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

model.save(f"YOLO/models/yolo11n-seg-{classification}-{modelversion}.pt")
