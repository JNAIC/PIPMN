import librosa
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch
import IPython.display as ipd
import random
import warnings
from spafe.features import pncc,gfcc,bfcc,mfcc,lfcc,lpc,ngcc,rplp,psrcc,msrcc
from cProfile import label
from itertools import dropwhile
import torch.nn as nn 
import torch
from tqdm import tqdm
from torchlibrosa import Spectrogram,LogmelFilterBank
from torchlibrosa import SpecAugmentation
import torchaudio
import random
from torch.utils.tensorboard import SummaryWriter
from Model.Main_Model import PIPMN
def train(data_path,random_state=42,epoches=3500,weight_decay=0.05,label_smoothing=0.1):
    path = data_path
    df = pd.read_csv(path+'/metadata/UrbanSound8K.csv')
    extracted = []
    for index_num,row in tqdm(df.iterrows()):
        warnings.filterwarnings('ignore')
        file_name = os.path.join(os.path.abspath(path),'fold'+str(row["fold"])+'/',str(row['slice_file_name'])) 
        final_class_labels = row['class']   
        feature=np.load(os.path.abspath(path)+"/Numpy_4_gfcc_bfcc_mfcc_lfcc_ngcc/"+'fold'+str(row["fold"])+'/'+str(row['slice_file_name']).split('.')[0]+'.npy')
        extracted.append([feature,final_class_labels])
    ext_df = pd.DataFrame(extracted,columns=['feature','class'])
    x_all = np.array(ext_df[['feature']].to_numpy())
    x=[]
    for i in x_all:
        #print(i[0].shape)
        x.append(i[0])
    y = np.array(ext_df['class'].tolist())
    le = LabelEncoder()
    y = le.fit_transform(y)
    y_all=np.eye[10](y)
    x_train, x_test, y_train, y_test = train_test_split(np.array(x), y_all, test_size=0.1, random_state = random_state)
    print(x_train[0].shape)
    print("Number of training samples = ", x_train.shape[0])
    print("Number of testing samples = ",x_test.shape[0])
    train_model=PIPMN(in_dim=100,num_classes=10,layer_dim_coef=[4,8],drop_rate=0.5,process_lenth=10).cuda()
    print("Number of Parameters: ",sum(i.nelement() for i in train_model.parameters()),0.05)
    optim=torch.optim.AdamW(train_model.parameters(),weight_decay=weight_decay)
    loss=nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    x_train=torch.tensor(x_train).cuda()
    y_train=torch.tensor(y_train).cuda()
    x_test=torch.tensor(x_test).cuda()
    y_test=torch.tensor(y_test).cuda()
    print(x_train.shape,y_train.shape)

    for epoch in  range(epoches) :
        corr=0
        count=0
        train_model.train()
        optim.zero_grad()
        pre=train_model(x_train)
        corr+=sum(torch.argmax(pre,dim=1).cpu().detach().numpy()==torch.argmax(y_train,dim=1).cpu().detach().numpy())
        count+=1
        losses=loss(pre,y_train)
        losses.backward()
        optim.step()
        print(corr,count)
        count=0
        print("train: ",corr/len(x_train))
        corr=0
        train_model.eval()
        test_pre=train_model(x_test)
        corr+=sum(torch.argmax(test_pre,dim=1).cpu().detach().numpy()==torch.argmax(y_test,dim=1).cpu().detach().numpy())
        print("validation dataet accuracy: ",corr/len(y_test))

