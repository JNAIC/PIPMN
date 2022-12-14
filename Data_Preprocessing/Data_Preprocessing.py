import librosa
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import random
import warnings
from spafe.features import pncc,gfcc,bfcc,mfcc,lfcc,lpc,ngcc,rplp,psrcc,msrcc
path = './UrbanSound8K'
df = pd.read_csv(path+'/metadata/UrbanSound8K.csv')
os.makedirs(os.path.abspath(path)+"/Numpy_5_gfcc_bfcc_mfcc_lfcc_ngcc")
for i in range(10):
    os.makedirs(os.path.abspath(path)+"/Numpy_5_gfcc_bfcc_mfcc_lfcc_ngcc/"+"fold"+str(i+1))
for index_num,row in tqdm(df.iterrows()):
    warnings.filterwarnings('ignore')
    file_name = os.path.join(os.path.abspath(path),'fold'+str(row["fold"])+'/',str(row['slice_file_name'])) 
    final_class_labels = row['class']   
    audio,sr=librosa.load(file_name)
    if audio.shape[0] > 88200:
        #print(2)
        max_audio_start = audio.shape[0] - 88200
        audio_start = random.randint(0, max_audio_start)
        audio = audio[audio_start : audio_start + 88200]
    else:
        #print(int(self.segment_length - audio.shape[0]))
        audio = librosa.util.pad_center(audio,88200)  
    feature=gfcc.gfcc(audio,fs=sr,num_ceps=20)
    feature1=bfcc.bfcc(audio,fs=sr,num_ceps=20)
    feature2=mfcc.mfcc(audio,fs=sr,num_ceps=20)
    feature3=pncc.pncc(audio,fs=sr,low_freq=0,high_freq=10000,num_ceps=20)
    feature3=lfcc.lfcc(audio,fs=sr,num_ceps=20)
    feature4=ngcc.ngcc(audio,fs=sr,num_ceps=20,low_freq=0,high_freq=10000)
    feature=np.concatenate([feature,feature1,feature2,feature3,feature4],axis=1)
    np.save(os.path.abspath(path)+"/Numpy_5_gfcc_bfcc_mfcc_lfcc_ngcc/"+'fold'+str(row["fold"])+'/'+str(row['slice_file_name']).split('.')[0]+'.npy',feature)
