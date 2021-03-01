# +++++++++++++++++++++++++++++++++++++++++++++DrivenData Competition+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -Author : Harish Agrawal
# -Neural Net : Inception-v3+lstm(1024 cells)
# -Using cross validation scheme 
# -Ideas that I want to try:
#                           1 - No training to CNN just train the lstm and last fc layer
#                           2 - Train CNN with less learning rate and see the results
#                           3 - Try with different CNN models from available and check which gives best results
#                           4 - First try different configurations of models as well as hidden parameters and once satisfied 
#                               with rate of drop of loss then go for training on large data (micro)
# -Don't judge by the count of correct prediction on validation data, instead submit your output on website to validate 
#   if it is doing properly or not.
# -Write code for generating output file as specified.
# -Find a way to segregate training data properly.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import print_function
from __future__ import division
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset , DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import datetime
import os
import copy
from tqdm import tqdm
import csv


from dLoader import *




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:{device}")
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("LSTM") != -1:
        print("LSTM Weight Initialize")
        # torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        print("Linear Weight Initialize")
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


# --------------------------------------------------------------------------------------------------------------------------------
# -------------------------------https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html---------------
# below code is taken from pytorch tutorials 

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained,progress=True, aux_logits=False,transform_input=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        # num_ftrs = model_ft.AuxLogits.fc.in_features
        # model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size




# -------------------------------------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------------------------------------
# --------------------------https://discuss.pytorch.org/t/cnn-lstm-implementation-for-video-classification/52018----------------
# below code is taken from the above webpage and modified as needed
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class VdoClassNet(nn.Module):
    def __init__(self,hidden_size,n_layers,dropt,bi,N_classes):
        super(VdoClassNet, self).__init__()

        self.hidden_size=hidden_size
        self.num_layers=n_layers
        self.n_cl=N_classes

        dim_feats = 2048
  
        self.cnn,input_size=initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
        self.cnn.fc=Identity()
        self.rnn = nn.LSTM(
            input_size=dim_feats,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropt,
            bidirectional=bi)
        # self.rnn.bias_ih.data.fill_(0) # initializing the rnn bias with zeros
        # self.rnn.bias_hh.data.fill_(0)
        if(bi==True):
            self.last_linear = nn.Linear(2*self.hidden_size,10)
            self.final_linear = nn.Linear(10,2)
        else:
            self.last_linear = nn.Linear(self.hidden_size,10)
            self.final_linear = nn.Linear(10,2)

        self.bce_loss = nn.BCELoss(reduction="sum")

            

    def forward(self, x, target=None):

        batch_size, timesteps, C,H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        
        c_out = self.cnn(c_in)

        r_out, (h_n, h_c) = self.rnn(c_out.view(-1,batch_size,c_out.size(-1)))  

        r_out2 = self.last_linear(r_out[-1])
        r_out3 = self.final_linear(r_out2)
        output = torch.sigmoid(r_out3)
        # print(output.size())
        # print(output.item())

        # output = output.argmax()
        if target is None:
            return 0,output
        else:
            target_l = FloatTensor(output.shape).fill_(0)
            index = range(target_l.size(0))
            target_l[index,target]=1
            # print(f"Target:{target} target_l:{target_l}")
            # print(f"Output:{output}")
            loss = self.bce_loss(output,target_l)
            # loss =0
            return loss,output

# # -------------------------------------------------------------------------------------------------------------------------------


# Initialize the model for this run
model_name = "inception"
num_classes = 2
feature_extract = True
# neuNet,ip=initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)


# print(neuNet)

neuNet = VdoClassNet(hidden_size=1024,n_layers=2,dropt=0,bi=True,N_classes=num_classes).to(device)
neuNet.apply(weights_init_normal)



# --------------------------------Dataset Loader-----------------------------------------------------------
microData=vdo_dataset(datapath="../data/micro/",type_labels=microtrainlabels)
print(f"Length of Training data :{len(microData)}")
microLen = len(microData)


# ------------------------------------End Dataset Loader-------------------------------------------------------

# -------------------------------------TrainLoop---------------------------------------------------------------------
epochs = 50
optimizer = torch.optim.Adam(neuNet.parameters(),lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                      patience=5, verbose=True)
file1 = open(r"logs1.txt", "w")

minLoss=99999999999
# neuNet.load_state_dict(torch.load("weights/FullLFinalWeights.pth"))
for epoch in range(epochs):

    trainSet,valSet = random_split(microData,[2000,microLen-2000])

    trainloader = torch.utils.data.DataLoader(
        trainSet, batch_size=1, shuffle=True, num_workers=4,pin_memory=True, collate_fn=microData.collate_fn
        )
    # print(f"trainloader length : {len(trainloader)}")

    valloader = torch.utils.data.DataLoader(
        valSet, batch_size=1, shuffle=True, num_workers=2,pin_memory=True, collate_fn=microData.collate_fn
        )
    # print(f"valloader length : {len(valloader)}")

    neuNet.train()
    Tol=0
    epoch_start_time = time.time()
    for k,(data,target) in enumerate((trainloader)):

        # plt.imshow(data[0][2].permute(1,2,0)/255)
        # plt.show()
        data = Variable(data.to(device))
        # cv2.imshow("t",data[0])
        # cv2.waitkey(0)
        # print(f"Size of Vdo data :{data.size()}")
        target = Variable(target.to(device), requires_grad=False)
        # print(target.size())
        # print(type(target[0]))
        # print(target.unsqueeze(0).size())
        # target=target.unsqueeze(0).unsqueeze(0)
        # print(type(target[0]))
        # target=transforms.ToTensor()(target)

        loss,output = neuNet(data,target)


        
        # print(f"Output:{output}")
        # print(output.size())
        # print(f"Target:{target}")
        # print(f"Loss:{loss}")

        file1.write(f"\n Epoch:{epoch} Batch:{k} Loss:{loss} Target:{target.data.tolist()} Output:{torch.argmax(output,dim=1).data.tolist()}")

        Tol +=loss.item()

        loss.backward()
        if k%4 == 0:
            optimizer.step()
            optimizer.zero_grad()

    epoch_time = datetime.timedelta(seconds=time.time() - epoch_start_time)
    print(f"epochs {epoch}/{epochs} epoch_loss : {Tol}  Epoch TIme {epoch_time}")

    if (epoch+1)%1==0:
        print("Evaluating Model on Validation Dataset")
        count = 0
        neuNet.eval()
        valloss = 0
        valid_start_time = time.time()
        for j,(valdata,valTar) in enumerate((valloader)):
            valdata = Variable(valdata.to(device))
            valTar = Variable(valTar.to(device), requires_grad=False)

            vloss,valoutput = neuNet(valdata,valTar)
            valloss+=vloss.item()
            
            #calculating number of correct predictions 
            # valoutput = torch.argmax(valoutput,dim=1)
            # indices = range(len(valTar))
            # correct = valTar[indices]==valoutput[indices]
            # count += sum(correct.int()).item()
            # print(f"count:{count}")

        valid_time = datetime.timedelta(seconds=time.time() - valid_start_time)
        print(f"Validation loss : {valloss} Validation Time {valid_time} " )
        scheduler.step(Tol)
        torch.save(neuNet.state_dict(), f"weights/Weights.pth")
        if minLoss >= valloss:
            minLoss = valloss
            print(f"Saving Weights of {epoch+1}th epoch")
            torch.save(neuNet.state_dict(), f"weights/Weights_{epoch+1}_{minLoss}.pth")

# file1.close()
print(f"Saving Weights of final epoch")
torch.save(neuNet.state_dict(), f"FinalWeights.pth")
