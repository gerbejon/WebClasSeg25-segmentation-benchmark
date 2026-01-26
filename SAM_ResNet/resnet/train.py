from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from datasets import load_dataset
import torch.nn as nn
from torchvision.models import resnet18
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
import sys
# dataset = load_dataset('imagefolder', data_dir='./webclasseg25-visual-fc-seg')
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__ ))))
sys.path.append(os.path.abspath(os.getcwd()))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__ ))))
from tools.parser import get_common_parser




class MaskedSegmentDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.samples = []
        self.class_to_idx = {}

        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.endswith(('.png', '.jpg')):
                    self.samples.append((os.path.join(class_dir, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label




class SegmentClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)



# segm = 'fc'
if __name__ == '__main__':
    parser = get_common_parser('Common parser')
    args = parser.parse_args()
    classification = args.classification
    dataset_folder = f"webclasseg25-visual-{classification}-seg-resnet"

    print("\n\nTraining segmentation dataset {}...\n\n".format(dataset_folder))

    # Load data
    train_dataset = MaskedSegmentDataset(root_dir=os.path.join('./SAM_ResNet/datasets', dataset_folder, 'train'))
    val_dataset = MaskedSegmentDataset(root_dir=os.path.join('./SAM_ResNet/datasets', dataset_folder, 'valid'), transform=train_dataset.transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Model
    num_classes = len(train_dataset.class_to_idx)
    model = SegmentClassifier(num_classes).cuda()

    # Optimizer
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        print(f"Epoch {epoch} Loss: {total_loss / len(train_loader):.4f}")

        # Validation (optional)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.cuda(), labels.cuda()
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"Val Acc: {correct / total:.2%}")

    torch.save(model, f"segment_classifier_full_model_{classification}.pt")
    # model = torch.load("./ResNet18/segment_classifier_full_model.pt", weights_only=False).cuda()
    # model.eval()
