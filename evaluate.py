import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from model import ResNet, block

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os

if os.path.exists('/content/drive/MyDrive/ml-data'):
    data_directory = '/content/drive/MyDrive/ml-data' 
else:
    data_directory = './dataset'

#defining the image transformation

import torchvision.transforms.functional as TF

class increase_contrast(object):
    def __call__(self, img):
        return TF.adjust_contrast(img, contrast_factor=1.8)

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    increase_contrast(),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root = data_directory,transform = None)
classes = dataset.classes
#train test validation

total = len(dataset)
train_size = int(0.8 * total)
val_size = int(0.1 * total)
test_size = total - train_size-val_size

train_set,val_set,test_set = random_split(
    dataset,
    [train_size,val_size,test_size],
    generator=torch.Generator().manual_seed(40)
)

class transform_subset(Dataset):
  def __init__(self,subset,transform):
    self.subset = subset
    self.transform = transform

  def __getitem__(self,index):
    img,label = self.subset[index]

    if self.transform:
      img = self.transform(img)

    return img,label

  def __len__(self):
    return len(self.subset)

modified_test = transform_subset(test_set,test_transform)
test_loader = DataLoader(modified_test,batch_size=32,shuffle=False)

def ResNet50(img_channels=3, num_classes = 5):
  return ResNet(block,[3,4,6,3],img_channels,num_classes)
model = ResNet50(img_channels=3, num_classes=5).to(device)

model.load_state_dict(torch.load('best_model.pth', map_location=device))

predict_list = []
label_list = []

model.eval()


with torch.no_grad():
  for data in test_loader:
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)

    #Forward pass and makes predictions
    outputs = model(images)
    _,predicted = torch.max(outputs,1)

    predict_list.extend(predicted.cpu().numpy())
    label_list.extend(labels.cpu().numpy())

prediction_arr = np.array(predict_list)
label_arr = np.array(label_list)

acc = accuracy_score(label_arr,prediction_arr)*100

rpt = classification_report(label_arr,prediction_arr,target_names=classes,digits=4)

print(f'Test Set Accuracy = {acc:.2f}%')
print('Classification Report')
print(rpt)

#generating confusion matrix

cm = confusion_matrix(label_arr, prediction_arr, labels=list(range(len(classes))))
plt.figure(figsize=(8, 6))

sns.heatmap(cm, annot = True, fmt = 'd',cmap='Blues', xticklabels = classes,yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
