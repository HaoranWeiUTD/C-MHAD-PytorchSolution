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
        self.fc2 = nn.Linear(256, 56)
        self.fc3 = nn.Linear(56, 7)
        self.dense2_bn = nn.BatchNorm1d(56)
        
        self.Vconv1 = nn.Conv3d(1, 16, (3,3,3), stride=(1,2,2), padding=(0,1,1))
        self.Vbn1 = nn.BatchNorm3d(16)
        self.Vconv2 = nn.Conv3d(16, 32, (3,3,3), stride=(1,1,2), padding=(1,1,1))
        self.Vbn2 = nn.BatchNorm3d(32)
        self.Vconv3 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.Vbn3 = nn.BatchNorm3d(64)
        self.Vconv4 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.Vbn4 = nn.BatchNorm3d(64)
        self.Vpool1 = nn.MaxPool3d(2, stride=2)      
        self.Vpool2 = nn.MaxPool3d((2,2,2), stride=(1, 2, 2)) 
        self.Vfc1 = nn.Linear(64 * 2 * 7 * 5, 128)
        self.Vdense1_bn1 = nn.BatchNorm1d(128)
      
    def forward(self, x, y):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.pool1(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 1 * 6)
        x = F.relu(self.dense1_bn(self.fc1(x)))
        
        y = self.Vpool1(F.relu(self.Vbn1(self.Vconv1(y))))
        y = self.Vpool1(F.relu(self.Vbn2(self.Vconv2(y))))
        y = self.Vpool1(F.relu(self.Vbn3(self.Vconv3(y))))
        y = self.Vpool1(F.relu(self.Vbn4(self.Vconv4(y))))
        y = y.view(-1, 64 * 2 * 7 * 5)
        y = F.relu(self.Vdense1_bn1(self.Vfc1(y)))
        #pdb.set_trace()       
        z = torch.cat((x, y), 1)
        z = F.relu(self.dense2_bn(self.fc2(z)))
        z = self.dr1(z)
        z = self.fc3(z)
        return z
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
