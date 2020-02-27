from torch.utils.data import Dataset
from .preprocess import *
import os
import xlrd
import pandas
import pdb
class CMHADDataset(Dataset):
    """BBC Lip Reading dataset."""

    def build_file_list(self, dir, set):
        labels = ['Action1','Action2','Action3','Action4','Action5','Action6','Action7']
        completeList = []
        subject = 1
        while subject<=12:
            subdir = dir+"/Subject"+str(subject)
            files = os.listdir(subdir)
            LabelPath = xlrd.open_workbook(subdir+"/ActionOfInterestTraSubject"+str(subject)+".xlsx")
            sheet = LabelPath.sheet_by_index(0) 
            #print(str(sheet.nrows)+" rows in Total")
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
                dirpath = subdir + "/InertialData/inertial_sub"+str(subject)+"_tr"+str(int(sheet.cell_value(m, 0)))+".csv"
                df = pandas.read_csv(dirpath)
                #print(df)
                MissFrames = 6005-len(df.index)
                midtime = sheet.cell_value(m, 3)/2 + sheet.cell_value(m, 2)/2
                #midframe = MissFrames+int(50*midtime)  #framerate = 50; starting from 0 fram, indicating 0.00 seconds.   
                if (set == "val") and (sheet.cell_value(m, 0) in valvideo) :
                    print("Creating Vallidation dataset for Action"+ str(int(sheet.cell_value(m, 1))), dirpath)
                    startframe = int(50*midtime) - 75 #150 frames in total
                    endframe = int(50*midtime) + 74
                    startframe, endframe = self.check_overflow(startframe, endframe)
                    entry = (int(sheet.cell_value(m, 1)-1), dirpath, startframe, startframe+149,MissFrames,subject)
                    completeList.append(entry)
                
                elif (set == "train") and (sheet.cell_value(m, 0) not in valvideo) :
                    print("Creating Training dataset for Action"+ str(int(sheet.cell_value(m, 1))), dirpath)
                    startframe = int(50*midtime) - 100 #200frames in total length, using only 150 frames from 200 as data augmentation
                    endframe = int(50*midtime) + 99
                    startframe, endframe = self.check_overflow(startframe, endframe)
                    for n in range(15):
                        #print(n)
                        entry = (int(sheet.cell_value(m, 1)-1), dirpath, startframe+3*n, startframe+3*n+149,MissFrames,subject)
                        completeList.append(entry)
            if set == "test":
                for o in valvideo:
                    startframe = MissFrames
                    dirpath = subdir + "/InertialData/inertial_sub"+str(subject)+"_tr"+str(o)+".csv"
                    print("Creating Testing dataset for", dirpath)
                    while startframe <= 5851:
                        entry = (0, dirpath, startframe, startframe+149, MissFrames,subject)
                        completeList.append(entry)            
                        startframe = startframe + 10
            subject = 1+subject           
        print("Size of data : " + str(len(completeList)))       
        print(completeList)        
        return labels, completeList

    def check_overflow(self, startframe, endframe):
        if startframe < 4: #avoid overflow
            endframe =  endframe+4-startframe
            startframe = 4
        elif endframe > 6000:
            startframe = startframe - (endframe-6000)
            endframe = 6000
        return startframe, endframe
           
    def __init__(self, directory, set, augment=True):
        self.label_list, self.file_list = self.build_file_list(directory, set)
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        label, filename, startframe, endframe, MissFrames,subject = self.file_list[idx]
        Inerframes = load_inertial(filename, startframe-MissFrames)
        sample = {'temporalvolume': Inerframes, 'label': torch.LongTensor([label]), 'MiddleTime':(startframe+75)/50, 'subject':subject}
        return sample
