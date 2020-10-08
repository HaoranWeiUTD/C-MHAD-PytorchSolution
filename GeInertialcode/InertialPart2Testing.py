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
import os, xlrd
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
TP_DetectionOnly = 0
TP_DetectionAndRecognition = 0
PredictPositive = 0
ActualPositive = 0
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
    if maxvalues[0]>0.95 and abs(MiddleTime[0]-TimeRestrict)>1.5:
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
                    ############ This part added at 2020/10/07 is used for F1 Calculation ########
                    resultdir = options["validation"]["dataset"]+"/Subject"+str(Lastsubject.numpy())
                    resultfiles = os.listdir(resultdir)
                    resultLabelPath = xlrd.open_workbook(resultdir+"/ActionOfInterestTraSubject"+str(Lastsubject.numpy())+".xlsx")
                    resultsheet = resultLabelPath.sheet_by_index(0)
                    for m in range(resultsheet.nrows):
                        if m==0:
                            continue
                        # Only consider video clip no.10 as testing data
                        if (resultsheet.cell_value(m, 0) == 10) and (LastMiddleTime > resultsheet.cell_value(m, 2)) and (LastMiddleTime < resultsheet.cell_value(m, 3)):
                            #pdb.set_trace()
                            TP_DetectionOnly = TP_DetectionOnly+1
                        if (resultsheet.cell_value(m, 0) == 10) and (LastMiddleTime > resultsheet.cell_value(m, 2)) and (LastMiddleTime < resultsheet.cell_value(m, 3)) and (resultsheet.cell_value(m, 1)==Lastmaxindices+1):
                            TP_DetectionAndRecognition = TP_DetectionAndRecognition+1                                             
                    #################################  Calculation Done ########################### 
                    PredictPositive  = PredictPositive+1
                    TimeRestrict =  LastMiddleTime
                    Lastmaxindices =-1
                    Lastmaxvalues = -1
                    count = 0
subject = 1
while subject<=12:
    subdir = options["validation"]["dataset"]+"/Subject"+str(subject)
    files = os.listdir(subdir)
    LabelPath = xlrd.open_workbook(subdir+"/ActionOfInterestTraSubject"+str(subject)+".xlsx")
    sheet = LabelPath.sheet_by_index(0) 
    for m in range(sheet.nrows):
        if m==0:
            continue
        if sheet.cell_value(m, 0) == 10:
            ActualPositive = ActualPositive+1
    subject = subject+1

with open(options["testing"]["resultfilelocation"], "a") as outputfile:           
                    outputfile.write("\n TP_DetectionOnly: {}, ActualPositive: {}, PredictPositive: {}, Recall: {}, Precision: {}, F1: {}" .format(TP_DetectionOnly, ActualPositive, PredictPositive, TP_DetectionOnly/ActualPositive, TP_DetectionOnly/PredictPositive, 2*(TP_DetectionOnly/ActualPositive)*(TP_DetectionOnly/PredictPositive)/(TP_DetectionOnly/ActualPositive+TP_DetectionOnly/PredictPositive)))
                    outputfile.write("\n TP_DetectionAndRecognition: {}, ActualPositive: {}, PredictPositive: {}, Recall: {}, Precision: {}, F1: {}" .format(TP_DetectionAndRecognition, ActualPositive, PredictPositive, TP_DetectionOnly/ActualPositive, TP_DetectionOnly/PredictPositive, 2*(TP_DetectionAndRecognition/ActualPositive)*(TP_DetectionAndRecognition/PredictPositive)/(TP_DetectionAndRecognition/ActualPositive+TP_DetectionAndRecognition/PredictPositive)))
ResultList = np.array(ResultList)
# save to csv file
np.savetxt('testscore.csv', ResultList, delimiter=',')