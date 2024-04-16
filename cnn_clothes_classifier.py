#install requirements
!pip install torchmetrics

#import libraries 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall

# Load datasets
from torchvision import datasets
import torchvision.transforms as transforms

train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

#Get number of classes
classes = train_data.classes
num_classes = len(train_data.classes)

print(classes, num_classes)

#Define variables
input_channels = 1
output_channels = 16
image_size = train_data[0][0].shape[1]

#Define classifier 
class MultiClassImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultiClassImageClassifier, self).__init__()
        # Convolutional layer
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        # ReLU activation layer
        self.relu = nn.ReLU()
        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Flatten layer
        self.flatten = nn.Flatten()
        # Fully connected layer
        self.fc = nn.Linear(output_channels * (image_size//2)**2, num_classes)
         
    def forward(self, x):
        # Convolutional layer
        x = self.conv1(x)
        # ReLU activation layer
        x = self.relu(x)
        # Max pooling layer
        x = self.maxpool(x)
        # Flatten the output of the convolutional layer
        x = self.flatten(x)
        # Fully connected layer
        x = self.fc(x)
        return x
    
#combine dataset and sampler 
dataloader_train = DataLoader(
    train_data,
    shuffle = True,
    batch_size = 10
)

#augmentations for clothes classification 
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAutocontrast(),
    transforms.Resize(image_size)
])

#define training function
def train_model(optimizer, net, num_epochs):
    num_processed = 0
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        running_loss = 0
        num_processed = 0
        for features, labels in dataloader_train:
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_processed += len(labels)
        print(f'epoch {epoch}, loss: {running_loss / num_processed}')
        
    train_loss = running_loss / len(dataloader_train)

# clothes classifier training loop
net = MultiClassImageClassifier(num_classes)
optimizer = optim.Adam(net.parameters(), lr = 0.001) 

#train model 
train_model(
    optimizer = optimizer,
    net = net,
    num_epochs = 2,
)

#test model on the test set 
dataloader_test = DataLoader(
    test_data,
    batch_size = 10,
    shuffle = False
)

#define metrics
#accuracy 
accuracy_metric = Accuracy(task='multiclass', num_classes = num_classes)
#precision 
precision_metric = Precision(task='multiclass', num_classes = num_classes, average = None)
#recall 
recall_metric = Recall(task='multiclass', num_classes = num_classes, average = None)

#run model on test set 
net.eval()
predicted = []
for i, (images, labels) in enumerate(dataloader_test):
    output = net.forward(images.reshape(-1, 1, image_size, image_size))
    cat = torch.argmax(output, dim = -1)
    predicted.extend(cat.tolist())
    accuracy_metric(cat, labels)
    precision_metric(cat, labels)
    recall_metric(cat, labels)

#compute metrics 
accuracy = accuracy_metric.compute().item()
precision = precision_metric.compute().tolist()
recall = recall_metric.compute().tolist()
print('Accuracy:', accuracy)
print('Precision (per class):', precision)
print('Recall (per class):', recall)