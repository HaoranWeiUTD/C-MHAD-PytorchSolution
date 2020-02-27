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
ResultList = []
for i_batch, sample_batched in enumerate(testdataloader):
    print(i_batch)
    input = Variable(sample_batched['temporalvolume'])
    labels = sample_batched['label']
    MiddleTime = sample_batched['MiddleTime']
    subject = sample_batched['subject']
    if(options["general"]["usecudnn"]):
        input = input.cuda(options["general"]["gpuid"])
        labels = labels.cuda(options["general"]["gpuid"])
    outputs = model(input)
    outputs = nn.Softmax(dim=1)(outputs)
    maxvalues, maxindices = torch.max(outputs.data, 1)
    ResultList.append([MiddleTime[0].tolist()]+outputs[0].data.cpu().tolist())
    if maxvalues[0]>0.98 and abs(MiddleTime[0]-TimeRestrict)>1.5:
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
                    outputfile.write("\nmaxvalues: {}, maxindices: {}, MiddleTime: {}, subject:{}, outputs: {}" .format(Lastmaxvalues, Lastmaxindices+1,LastMiddleTime,Lastsubject, Lastoutputs))
                    TimeRestrict =  LastMiddleTime
                    Lastmaxindices =-1
                    Lastmaxvalues = -1
                    count = 0

#pdb.set_trace()
ResultList = np.array(ResultList)
# save to csv file
np.savetxt('testscore.csv', ResultList, delimiter=',')

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
    input = Variable(sample_batched['temporalvolume'])
    labels = sample_batched['label']
    if(options["general"]["usecudnn"]):
        input = input.cuda(options["general"]["gpuid"])
        labels = labels.cuda(options["general"]["gpuid"])
    outputs = model(input)
    count += validate(outputs, labels)
    print(count)
    accuracy = count / len(validationdataset)
with open(options["testing"]["resultfilelocation"], "a") as outputfile:
    outputfile.write("\ncorrect count: {}, total count: {} accuracy: {}" .format(count, len(validationdataset), accuracy ))
'''