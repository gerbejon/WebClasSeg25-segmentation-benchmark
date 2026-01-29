import torch
from PIL import Image
import os
import sys
# sys.path.append(os.path.abspath(os.getcwd()))
# from SAM_ResNet.resnet.train import SegmentClassifier
from torchvision import transforms





class ResNet18_segment_classifier:
    def __init__(self, segm, model_dir=None):
        if model_dir is None:
            model_dir = f"./SAM_ResNet/resnet/segment_classifier_full_model_{segm}.pt"
        self.model = torch.load(model_dir, weights_only=False).cuda()
        self.model.eval()
        self.segm = segm
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    def preproc_image(self, image: Image):
        input_batch = self.transform(image).unsqueeze(0).cuda()
        return input_batch

    def predict(self, img_dir=None, image=None):
        if image is None:
            if img_dir is None:
                img_dir = f'/home/ubuntu/dataset_benchmark/ResNet18/webclasseg25-visual-{self.segm}-seg/test/1/10148.png'
            image = Image.open(img_dir).convert('RGB')
        input_batch = self.preproc_image(image)

        with torch.no_grad():
            output = self.model(input_batch)
        predicted_class = output.argmax(dim=1).item()
        print(f"Predicted class: {predicted_class}")
        return predicted_class

    def predict_batch(self, img_dirs:list[str] = None, folder_dir:str = None):
        if img_dirs is None:
            if folder_dir is None:
                folder_dir = f'/home/ubuntu/dataset_benchmark/ResNet18/webclasseg25-visual-{self.segm}-seg/valid/5'
            img_dirs = [os.path.join(folder_dir, img) for img in os.listdir(folder_dir)]
        input_batch = torch.stack([self.transform(Image.open(img).convert('RGB')).to("cuda") for img in img_dirs if img.endswith('.png')])
        with torch.no_grad():
            output = self.model(input_batch)
        return output.argmax(dim=1).tolist()

if __name__ == '__main__':
    # model = torch.load("./segment_classifier_full_model.pt", weights_only=False).cuda()
    # model.eval()
    example_image = '/home/ubuntu/dataset_benchmark/ResNet18/webclasseg25-visual-fc-seg/test/1/10148.png'
    model = ResNet18_segment_classifier(segm='fc')
    # model.predict(example_image)
    folder_dir = '/home/ubuntu/dataset_benchmark/ResNet18/webclasseg25-visual-fc-seg/valid/7'
    res = model.predict_batch(folder_dir=folder_dir)

