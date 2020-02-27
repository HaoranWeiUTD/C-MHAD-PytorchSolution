from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import CMHADDataset
from torch.utils.data import DataLoader
import os

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
            input_x = Variable(sample_batched['temporalvolume_x'])
            input_y = Variable(sample_batched['temporalvolume_y'])
            labels = sample_batched['label']
            if(self.usecudnn):
                input_x = input_x.cuda(self.gpuid)
                input_y = input_y.cuda(self.gpuid)
                labels = labels.cuda(self.gpuid)

            outputs = model(input_x,input_y)

            count += self.validate(outputs, labels)

            print(count)


        accuracy = count / len(self.validationdataset)
        with open(self.accuracyfilelocation, "a") as outputfile:
            outputfile.write("\ncorrect count: {}, total count: {} accuracy: {}" .format(count, len(self.validationdataset), accuracy ))

    def validate(self, modelOutput, labels):
        maxvalues, maxindices = torch.max(modelOutput.data, 1)

        count = 0

        for i in range(0, labels.squeeze(1).size(0)):

            if maxindices[i] == labels.squeeze(1)[i]:
                count += 1

        return count
