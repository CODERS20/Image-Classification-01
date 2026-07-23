import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets

from torch.utils.data import Dataset, DataLoader, random_split

from model import ResNet, block

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


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    increase_contrast(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=180),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    increase_contrast(),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root = data_directory,transform = None)

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

modified_train = transform_subset(train_set,train_transform)
modified_val = transform_subset(val_set,test_transform)
modified_test = transform_subset(test_set,test_transform)

train_loader = DataLoader(modified_train,batch_size=32,shuffle=True,num_workers=2,pin_memory=True)
val_loader = DataLoader(modified_val,batch_size=32,shuffle=False)
test_loader = DataLoader(modified_test,batch_size=32,shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#creating the model

def ResNet50(img_channels=3, num_classes = 5):
  return ResNet(block,[3,4,6,3],img_channels,num_classes)

model = ResNet50(img_channels=3, num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

#adjusts the learning rate to prevent bouncing according the losses
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)

#training
best_val_loss = float('inf')

for epoch in range(50):
    print(f'Training epoch {epoch}..')
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)

    model.eval()
    val_running_loss = 0.0

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

    val_loss = val_running_loss / len(val_loader)

    print(f'Train Loss: {epoch_loss:.4f} | Validation Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"New best val loss, saved.")

    scheduler.step(val_loss)

print("Training complete. Best weights saved to best_model.pth")
