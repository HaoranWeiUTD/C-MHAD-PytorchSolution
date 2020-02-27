from torch.utils.data import Dataset
from .preprocess import *
import os
import xlrd

class CMHADDataset(Dataset):
    """BBC Lip Reading dataset."""

    def build_file_list(self, dir, set):
        labels = ['Action1','Action2','Action3','Action4','Action5','Action6','Action7']
        completeList = []
        subject = 1      
        while subject<=12:
            subdir = dir+"/Subject"+str(subject)
            LabelPath = xlrd.open_workbook(subdir+"/ActionOfInterestTraSubject"+str(subject)+".xlsx")
            sheet = LabelPath.sheet_by_index(0) 
            min = 2
            max = 3
            for l in range(sheet.nrows-1): 
                val = sheet.cell_value(l+1, 3)-sheet.cell_value(l+1, 2)
                if val < min:
                    min = val
                if val > max:
                    max = val
            print("The Minimum action duration of this Subject is: "+str(min)+" seconds")
            print("The Maximum action duration of this Subject is: "+str(max)+" seconds")
            valvideo = [10]
            print("Validation Video include:", str(valvideo[0]))

            for m in range(sheet.nrows):
                if m==0:
                    continue
                dirpath = subdir + "/VideoData/video_sub"+str(subject)+"_tr"+str(int(sheet.cell_value(m, 0)))+".avi"
                midtime = sheet.cell_value(m, 3)/2 + sheet.cell_value(m, 2)/2

                midframe = int(15*midtime)  #framerate = 15; starting from 0 fram, indicating 0.00 seconds.   
                if (set == "val") and (sheet.cell_value(m, 0) in valvideo) :
                    print("Creating Vallidation dataset for Action"+ str(int(sheet.cell_value(m, 1))), dirpath)
                    startframe = int(15*midtime) - 22 #45 frames in total
                    endframe = int(15*midtime) + 22
                    startframe, endframe = self.check_overflow(startframe, endframe)
                    entry = (int(sheet.cell_value(m, 1)-1), dirpath, startframe, startframe+44,subject)
                    completeList.append(entry)
                
                elif (set == "train") and (sheet.cell_value(m, 0) not in valvideo) :
                    print("Creating Training dataset for Action"+ str(int(sheet.cell_value(m, 1))), dirpath)
                    startframe = int(15*midtime) - 30 #61frames in total length, using only 45 frames from 60 as data augmentation
                    endframe = int(15*midtime) + 30
                    startframe, endframe = self.check_overflow(startframe, endframe)
                    for n in range(16):

                        entry = (int(sheet.cell_value(m, 1)-1), dirpath, startframe+n, startframe+44+n,subject)
                        completeList.append(entry)
            if set == "test":
                for o in valvideo:
                    startframe = 0
                    dirpath = subdir + "/VideoData/video_sub"+str(subject)+"_tr"+str(o)+".avi"
                    print("Creating Testing dataset for", dirpath)
                    while startframe <= 1756:
                        entry = (0, dirpath, startframe, startframe+44,subject)
                        completeList.append(entry)
                        startframe = startframe + 3
            subject = subject+1            
        print("Size of data : " + str(len(completeList)))            
        return labels, completeList

    def check_overflow(self, startframe, endframe):
        if startframe < 0: #avoid overflow
            endframe = endframe - startframe
            startframe = 0
        elif endframe > 1800 :
            startframe = startframe - (endframe-1800)
            endframe = 1800
        return startframe, endframe
           
    def __init__(self, directory, set, augment=True):
        self.label_list, self.file_list = self.build_file_list(directory, set)
        self.augment = augment
        print(self.label_list)
        print(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        #load video into a tensor
        label, filename, startframe, endframe,subject = self.file_list[idx]
        vidframes = load_video(filename, startframe)
        temporalvolume = bbc(vidframes, self.augment)

        sample = {'temporalvolume': temporalvolume, 'label': torch.LongTensor([label]), 'MiddleTime':(startframe+22)/15, 'subject':subject}

        return sample
