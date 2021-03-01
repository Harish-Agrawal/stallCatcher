# +++++++++++++++++++++++++++++++++++++++++++++DrivenData Competition+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -Author : Harish Agrawal
# -Data Loader : data loader for loading video data 
#  Need to do pre-processing before supplying it to the neural network.
#      - First find the region of interest and crop it so that computations will be less
#        while doing pre-processing like normalizing and handling 4-d tensor
#      - 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np  
import pandas as pd
from tqdm import tqdm                                                                                                                                                     
import cv2                                                                                                                                                               
import os, sys    
import matplotlib.pyplot as plt  
from torchvision import transforms 

from torch.utils.data import Dataset , DataLoader
import torch.nn.functional as F

import torchvision.io      
from torchvision.utils import save_image
import torch
import torchvision                                                                                                                                              
                                      
threshold = [[104,255], [13 ,143], [0 ,98]]                                                                                                                                                                         

train = pd.read_csv("../train_metadata.csv", sep=",")
# print(train.head())
# print(len(train))

trainlabels = pd.read_csv("../train_labels.csv", sep=",")
# print(len(trainlabels)) # 573048

test = pd.read_csv("../test_metadata.csv", sep=",")
# print(len(test)) # 14160
# print(test.head())


assert len(train) == len(trainlabels), "lengths of trainlabels.csv and train_metadata.csv differ"

nanotrain = train.loc[train['nano'] == True]
microtrain = train.loc[train['micro'] == True]
# print(len(nanotrain)) # 1413

nanotrainlabels = trainlabels.loc[train['nano'] == True]
microtrainlabels = trainlabels.loc[train['micro'] == True]
# print(len(nanotrainlabels) == len(nanotrain))
# print(nanotrainlabels.head())
# print(nanotrainlabels['filename'].head())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resize(image, size):
    # image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #-- changed the above interpolation from nearest to bilinear
    #-- this function will take 4-d input and give back 4-d output
    image = F.interpolate(image, size=size, mode="bilinear", align_corners = False )
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    return image


# Dataloader: while training pass the train_flag as True and for testing False
#             When train_flag = True --- Output is vdo and label(groundTruth which is to be learnt)
#                             = False--- Output is vdo and vdo name  
class vdo_dataset(Dataset):
    def __init__(self,datapath="../data/nano/",type_labels=nanotrainlabels,train_flag=True,length=310,sizeFrame=299):
        self.datapath = datapath
        self.type_labels = type_labels
        self.vdo_names = np.stack([c for n, c in self.type_labels['filename'].items()], axis=0) #tpye: numpy.ndarray
        self.length = length
        self.train_flag = train_flag
        self.sizeFrame = sizeFrame
        if self.train_flag:
            self.target_labels = np.stack([c for n, c in self.type_labels['stalled'].items()], axis=0) #tpye: numpy.ndarray

    def __getitem__(self, index):
        # creating path of vdo file to be read
        vdo_path = self.datapath+self.vdo_names[index]

        # reading the vdo and storing it into 4-d tensor [timestamp,channels,height,width]
        self.frames=torchvision.io.read_video(vdo_path)#,pts_unit='sec')
        frames = self.frames[0]

        # finding region of interest
        frame1 = frames[0]
        points = np.where((frame1[:,:,0] >= threshold[0][0])& (frame1[:,:,0] <= threshold[0][1])&\
            (frame1[:,:,1] >= threshold[1][0])& (frame1[:,:,1] <= threshold[1][1])&\
            (frame1[:,:,2] >= threshold[2][0])& (frame1[:,:,2] <= threshold[2][1]))

        # calculating box surrounding roi
        p2 = zip(points[0], points[1]) 
        p2 = [p for p in p2]                                                                                                                                                 
        rect = cv2.boundingRect(np.float32(p2))

        # co-ordinates of box for roi
        w1=rect[1]
        w2=rect[1]+rect[3]
        h1=rect[0]
        h2=rect[0]+rect[2]

        # Taking only ROI 
        frames = frames[:,h1:h2,w1:w2,:].permute(0,3,1,2).contiguous().type(torch.FloatTensor)

# ---------------------------------------------------------------------------------------------------------
        frames = frames/255
        preprocess = transforms.Compose([
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        frames = torch.stack([preprocess(frame) for frame in frames])

        # Inception-v3 requires input size as 299x299
        frames=resize(frames,self.sizeFrame)
        tmp_len=frames.size(0)

        # If video is too large ... take only specified number of frames
        # I am taking frames from middle of the videos as thought behind this is main content
        # i.e. if it is stalled or not will be in the middle of the videos not at the start 
        # nor at the end 
        if tmp_len > self.length:
            diff = tmp_len-self.length
            start = diff // 2
            frames = frames[start:start+self.length]

        targetL = None
        if self.train_flag:
            targetL = self.target_labels[index]
            return frames,targetL
        else:
            return frames,self.vdo_names[index]



    def collate_fn(self, batch):
        vdos, vdo_targets = list(zip(*batch))

        vdos = torch.stack(vdos)
        # vdo_targets = torch.cat(vdo_targets) 
        vdo_targets = [c for c in vdo_targets if c is not None]
        if self.train_flag:
            vdo_targets = torch.LongTensor(vdo_targets)
        return vdos,vdo_targets
     
    def __len__(self):
        return len(self.vdo_names)





