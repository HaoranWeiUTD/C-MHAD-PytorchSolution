from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
from data import CMHADDataset
from torch.utils.data import DataLoader
import os
import pdb
class Validator():
    def __init__(self, options):

        self.validationdataset = CMHADDataset(options["validation"]["dataset"],"val", False)
        self.validationdataloader = DataLoader(
                                    self.validationdataset,
                                    batch_size=options["input"]["batchsize"],
                                    shuffle=options["input"]["shuffle"],
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=True
                                )
        self.usecudnn = options["general"]["usecudnn"]

        self.batchsize = options["input"]["batchsize"]

        self.statsfrequency = options["training"]["statsfrequency"]

        self.gpuid = options["general"]["gpuid"]
        self.accuracyfilelocation = options["validation"]["accuracyfilelocation"]
    def epoch(self, model):
        print("Starting validation...")
        count = 0

        for i_batch, sample_batched in enumerate(self.validationdataloader):
            input = Variable(sample_batched['temporalvolume'])
            labels = sample_batched['label']
            if i_batch==0:
                import cv2
                import numpy as np
                print(input.size())
                print((input[0,:,0,:,:].permute(1, 2, 0).numpy()+5)*10)
                cv2.imwrite('CheckImage.jpg', (input[0,:,0,:,:].permute(1, 2, 0).numpy()+3)*40) 
                
            if(self.usecudnn):
                input = input.cuda(self.gpuid)
                labels = labels.cuda(self.gpuid)

            outputs = model(input)

            count += self.validate(outputs, labels)
            print(count)
            #pdb.set_trace()

        accuracy = count / len(self.validationdataset)
        with open(self.accuracyfilelocation, "a") as outputfile:
            outputfile.write("\ncorrect count: {}, total count: {} accuracy: {}" .format(count, len(self.validationdataset), accuracy ))

    def validate(self, modelOutput, labels):
        outputs = nn.Softmax(dim=1)(modelOutput)
        maxvalues, maxindices = torch.max(outputs.data, 1)
        
        print("Action",maxindices+1,"score",maxvalues)
        count = 0

        for i in range(0, labels.squeeze(1).size(0)):

            if maxindices[i] == labels.squeeze(1)[i]:
                count += 1

        return count
