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
                Idirpath = subdir + "/InertialData/inertial_sub"+str(subject)+"_tr"+str(int(sheet.cell_value(m, 0)))+".csv"
                Vdirpath = subdir + "/VideoData/video_sub"+str(subject)+"_tr"+str(int(sheet.cell_value(m, 0)))+".avi"
                df = pandas.read_csv(Idirpath)
                MissFrames = 6005-len(df.index)
                midtime = sheet.cell_value(m, 3)/2 + sheet.cell_value(m, 2)/2
                if (set == "val") and (sheet.cell_value(m, 0) in valvideo) :
                    print("Creating Vallidation dataset for Action"+ str(int(sheet.cell_value(m, 1))), "subject: ",str(subject))
                    startframe = int(50*midtime) - 75 #150 frames in total
                    endframe = int(50*midtime) + 74
                    startframe, endframe = self.check_overflow(startframe, endframe)

                    Vstartframe = int(15*midtime) - 22 #45 frames in total
                    Vendframe = int(15*midtime) + 22
                    Vstartframe, Vendframe = self.Vcheck_overflow(Vstartframe, Vendframe)
                    
                    entry = (int(sheet.cell_value(m, 1)-1), Idirpath, startframe, startframe+149,MissFrames,Vdirpath,Vstartframe,Vstartframe+44,subject)
                    completeList.append(entry)
                
                elif (set == "train") and (sheet.cell_value(m, 0) not in valvideo) :
                    print("Creating Training dataset for Action"+ str(int(sheet.cell_value(m, 1))), "subject: ",str(subject))
                    startframe = int(50*midtime) - 100 #200frames in total length, using only 150 frames from 200 as data augmentation
                    endframe = int(50*midtime) + 99
                    startframe, endframe = self.check_overflow(startframe, endframe)
                    
                    Vstartframe = int(15*midtime) - 30 #45 frames in total
                    Vendframe = int(15*midtime) + 29
                    Vstartframe, Vendframe = self.Vcheck_overflow(Vstartframe, Vendframe)
                                        
                    for n in range(15):
                        entry = (int(sheet.cell_value(m, 1)-1), Idirpath, startframe+3*n, startframe+3*n+149,MissFrames,Vdirpath,Vstartframe+n,Vstartframe+n+44,subject)
                        completeList.append(entry)
            if set == "test":
                for o in valvideo:
                    startframe = MissFrames
                    midtime = (MissFrames+75)/50
                    Vstartframe = int(15*midtime) - 22
                    Idirpath = subdir + "/InertialData/inertial_sub"+str(subject)+"_tr"+str(o)+".csv"
                    Vdirpath = subdir + "/VideoData/video_sub"+str(subject)+"_tr"+str(o)+".avi"
                    print("Creating Testing dataset for", "subject: ",str(subject))
                    while startframe <= 5851 and Vstartframe <= 1756:
                        entry = (0, Idirpath, startframe, startframe+149, MissFrames,Vdirpath, Vstartframe, Vstartframe+44,subject)
                        completeList.append(entry)            
                        startframe = startframe + 10
                        Vstartframe = Vstartframe + 3
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
        
    def Vcheck_overflow(self, startframe, endframe):
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

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        label, filename, startframe, endframe, MissFrames, Vdirpath, Vstartframe, Vendframe, subject = self.file_list[idx]
        Inerframes = load_inertial(filename, startframe-MissFrames)
        vidframes = load_video(Vdirpath, Vstartframe)
        temporalvolume = bbc(vidframes, self.augment)
        sample = {'temporalvolume_x': Inerframes, 'temporalvolume_y': temporalvolume, 'label': torch.LongTensor([label]), 'MiddleTime':(startframe+75)/50, 'subject':subject}
        return sample
