from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import toml
import cv2
from training import Trainer
from validation import Validator
import pdb
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
        self.conv1 = nn.Conv3d(1, 16, (3,3,3), stride=(1,2,2), padding=(0,1,1))
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, (3,3,3), stride=(1,1,2), padding=(1,1,1))
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(2, stride=2)      
        self.pool2 = nn.MaxPool3d((2,2,2), stride=(1, 2, 2)) 
        self.fc1 = nn.Linear(64 * 2 * 7 * 5, 128)
        self.dense1_bn1 = nn.BatchNorm1d(128)
        #self.fc2 = nn.Linear(512, 64)
        #self.dense1_bn2 = nn.BatchNorm1d(64)
        self.dr1 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, 7)
      
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.pool1(F.relu(self.bn3(self.conv3(x))))
        x = self.pool1(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 64 * 2 * 7 * 5)
        x = F.relu(self.dense1_bn1(self.fc1(x)))
        #pdb.set_trace()
        x = self.dr1(x)
        x = self.fc3(x)
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
