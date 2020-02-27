from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import toml
import cv2
from training import Trainer
from validation import Validator
from datetime import datetime, timedelta
from data import CMHADDataset
from torch.utils.data import DataLoader
import pdb
import numpy as np
print("Loading options...")
with open('options.toml', 'r') as optionsFile:
    options = toml.loads(optionsFile.read())

if(options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
    print("Running cudnn benchmark...")
    torch.backends.cudnn.benchmark = True

#load the model.
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
        #self.fc2 = nn.Linear(256, 56)
        self.fc2 = nn.Linear(256, 7)
        #self.dense2_bn = nn.BatchNorm1d(56)
        
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
        #z = F.relu(self.dense2_bn(self.fc2(z)))
        z = self.dr1(z)
        z = self.fc2(z)
        return z
model= Net()

print(options["general"]["modelsavepath"])
model.load_state_dict(torch.load(options["general"]["modelsavepath"]))
model.eval()
print(model)
#Move the model to the GPU.
if(options["general"]["usecudnn"]):
    model = model.cuda(options["general"]["gpuid"])
    

#load Testing model.
testdataset = CMHADDataset(options["validation"]["dataset"],"test", False)
testdataloader = DataLoader(
                                    testdataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=True
                                )
#Testing and output to 
print("Starting Testing...")
TimeRestrict = 0
Lastmaxindices = -1
Lastmaxvalues = -1
count = 0
for i_batch, sample_batched in enumerate(testdataloader):
    print(i_batch)
    input_x = Variable(sample_batched['temporalvolume_x'])
    input_y = Variable(sample_batched['temporalvolume_y'])
    labels = sample_batched['label']
    MiddleTime = sample_batched['MiddleTime']
    subject = sample_batched['subject']
    if(options["general"]["usecudnn"]):
        input_x = input_x.cuda(options["general"]["gpuid"])
        input_y = input_y.cuda(options["general"]["gpuid"])
        labels = labels.cuda(options["general"]["gpuid"])
    outputs = model(input_x,input_y)
    outputs = nn.Softmax(dim=1)(outputs)
    maxvalues, maxindices = torch.max(outputs.data, 1)
    if maxvalues[0]>0.987 and abs(MiddleTime[0]-TimeRestrict)>1.5:
        if Lastmaxindices == -1:
                Lastmaxindices = maxindices[0]
                Lastmaxvalues = maxvalues[0]
                Lastoutputs = np.asarray(outputs[0].data.cpu())
                LastMiddleTime = MiddleTime[0]   
                Lastsubject  = subject[0]              
        if maxindices[0] == Lastmaxindices and maxvalues[0] > Lastmaxvalues:
                Lastmaxvalues = maxvalues[0]
                Lastoutputs = np.asarray(outputs[0].data.cpu())
                LastMiddleTime = MiddleTime[0]
                Lastsubject  = subject[0]  
                count = 0
        else:
            if count < 3:
                count += 1
            else:
                with open(options["testing"]["resultfilelocation"], "a") as outputfile:           
                    outputfile.write("\nmaxvalues: {}, maxindices: {}, outputs: {}, MiddleTime: {}, subject:{}" .format(Lastmaxvalues, Lastmaxindices+1, Lastoutputs,LastMiddleTime,Lastsubject))
                    TimeRestrict =  LastMiddleTime
                    Lastmaxindices =-1
                    Lastmaxvalues = -1
                    count = 0
'''

def validate(modelOutput, labels):
    maxvalues, maxindices = torch.max(modelOutput.data, 1)
    count = 0
    for i in range(0, labels.squeeze(1).size(0)):
        if maxindices[i] == labels.squeeze(1)[i]:
            count += 1
    return count
    
validationdataset = CMHADDataset(options["validation"]["dataset"],"val", False)
validationdataloader = DataLoader(
                            validationdataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=options["input"]["numworkers"],
                            drop_last=False
                        )
count = 0

for i_batch, sample_batched in enumerate(validationdataloader):
    input_x = Variable(sample_batched['temporalvolume_x'])
    input_y = Variable(sample_batched['temporalvolume_y'])
    labels = sample_batched['label']
    if(options["general"]["usecudnn"]):
        input_x = input_x.cuda(options["general"]["gpuid"])
        input_y = input_y.cuda(options["general"]["gpuid"])
        labels = labels.cuda(options["general"]["gpuid"])
    outputs = model(input_x,input_y)
    count += validate(outputs, labels)
    print(count)
    accuracy = count / len(validationdataset)
with open(options["testing"]["resultfilelocation"], "a") as outputfile:
    outputfile.write("\ncorrect count: {}, total count: {} accuracy: {}" .format(count, len(validationdataset), accuracy ))
'''
