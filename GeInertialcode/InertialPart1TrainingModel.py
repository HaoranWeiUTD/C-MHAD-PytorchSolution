from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import toml
import cv2
from training import Trainer
from validation import Validator

print("Loading options...")
with open('options.toml', 'r') as optionsFile:
    options = toml.loads(optionsFile.read())

if(options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
    print("Running cudnn benchmark...")
    torch.backends.cudnn.benchmark = True

#Create the model.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, stride=(2, 2))      
        
        self.fc1 = nn.Linear(64 * 1 * 6, 128)
        self.dense1_bn = nn.BatchNorm1d(128)
        self.dr1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 7)
      
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.pool1(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 1 * 6)
        x = F.relu(self.dense1_bn(self.fc1(x)))
        x = self.dr1(x)
        x = self.fc2(x)
        return x

model= Net()

#Move the model to the GPU.
if(options["general"]["usecudnn"]):
    model = model.cuda(options["general"]["gpuid"])

trainer = Trainer(options)
validator = Validator(options)

for epoch in range(options["training"]["epoch"]):
    print("epoch Number: ",epoch )
    trainer.epoch(model, epoch)
    validator.epoch(model)
