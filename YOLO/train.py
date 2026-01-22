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
output_dir = args.output_dir if args.output_dir is not None else f"./YOLO/results_{classification}_{modelversion}"
epochs = args.epochs if args.epochs is not None else 200


# segm = sys.argv[1]
# version = 'v1'

# Load a pretrained YOLO11n model
model = YOLO("yolo11n-seg.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    # data="coco.yaml",  # Path to dataset configuration file
    data=f'./YOLO/webclasseg25-visual-{classification}-seg.yaml',
    epochs=epochs,  # Number of training epochs
    imgsz=512,  # Image size for training
    device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

model.save(f"yolo11n-seg-{classification}-{modelversion}.pt")
# Evaluate the model's performance on the validation set
# metrics = model.val()
#
# # Perform object detection on an image
# results = model("path/to/image.jpg")  # Predict on an image
# results[0].show()  # Display results
#
# # Export the model to ONNX format for deployment
# path = model.export(format="onnx")  # Returns the path to the exported model

# dataset_dir = f'./datasets_yolo/webclasseg25-visual-{segm}/images/test/'
# results = model(
#     [
#         dataset_dir + "66ffde79306dfe2088fd9ff9.jpg",
#         dataset_dir + "66ffde79306dfe2088fd9ffb.jpg"
#     ]
# )  # return a list of Results objects
#
# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     # result.save(filename="result.jpg")  # save to disk