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

class feedforward(nn.Module):
    def __init__(self,in_dim,out_dim,drop_rate=0.5,real_dim=10) -> None:
        super(feedforward,self).__init__()
        self.main=nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim,out_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(out_dim,in_dim),
            nn.Dropout(drop_rate),
        )
        self.conv=nn.Conv1d(kernel_size=3,in_channels=real_dim,out_channels=real_dim,padding=1,stride=1,groups=real_dim)
        self.fuse=nn.Sequential(
            nn.Linear(3*in_dim,in_dim),
            nn.Dropout(drop_rate),
        )
    def forward(self,x):
        #print(x.shape)
        x=x.permute(0,2,1)
        proj=self.conv(x)
        #print(proj.shape,x.shape)
        pre=x
        x=self.main(x)+x
        return self.fuse(torch.concat([x,pre,proj],dim=2)).permute(0,2,1)
class DenseMLPBlock(nn.Module):
    def __init__(self,in_dim,out_dim,drop_rate=0.5,layer_scale_init=1e-6,lenth=10,alpha=2) -> None:
        super(DenseMLPBlock,self).__init__()
        self.spa=feedforward(lenth,lenth*alpha,drop_rate=drop_rate,real_dim=in_dim)
        self.li=nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim,out_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),)
        self.gamma=nn.Parameter(layer_scale_init*torch.ones(out_dim),requires_grad=True)
        self.gamma_fi=nn.Parameter(layer_scale_init*torch.ones(in_dim))
        self.large=nn.Sequential(
            nn.Linear(in_dim,out_dim,bias=False),
            nn.Dropout(drop_rate),
        )
    def forward(self,x):
        #print(x.shape)
        x=self.gamma_fi*(self.spa(x))+x
        return self.gamma*(self.li(x))
class torch_net_5(nn.Module):
    def __init__(self,in_dim=50,num_classes=10,drop_rate=0.5,layer_dim_coef=[4,8],init_layer_scale=1e-6,num_heads=4,process_lenth=1):
        super(torch_net_5,self).__init__()
        self.first_in=nn.ModuleList()
        self.first_out=nn.ModuleList()
        self.layer_parameter_in=nn.ParameterList()
        self.layer_parameter_out=nn.ParameterList()
        self.coef=layer_dim_coef.copy()
        self.in_dim=in_dim
        self.small=nn.AdaptiveAvgPool1d(process_lenth)
        final_lenth=process_lenth
        for i in range(len(layer_dim_coef)):
            layer_dim_coef[i]*=in_dim
            print(layer_dim_coef[i])
           
        for i in range(len(layer_dim_coef)):
            if(i==0):
                self.first_in.append(DenseMLPBlock(in_dim,layer_dim_coef[i],drop_rate=drop_rate,lenth=final_lenth))
                self.layer_parameter_in.append(nn.Parameter(init_layer_scale*torch.ones(in_dim)))
            else:
                self.first_in.append(DenseMLPBlock(layer_dim_coef[i-1],layer_dim_coef[i],drop_rate=drop_rate,lenth=final_lenth))
                self.layer_parameter_in.append(nn.Parameter(init_layer_scale*torch.ones(layer_dim_coef[i-1])))
        for i in range(len(layer_dim_coef)-1,-1,-1):
            if(i==0):
                self.first_out.append(DenseMLPBlock(layer_dim_coef[i],in_dim,drop_rate=drop_rate,lenth=final_lenth))
                self.layer_parameter_out.append(nn.Parameter(init_layer_scale*torch.ones(layer_dim_coef[i])))
            else:
                self.first_out.append(DenseMLPBlock(layer_dim_coef[i],layer_dim_coef[i-1],drop_rate=drop_rate,lenth=final_lenth))
                self.layer_parameter_out.append(nn.Parameter(init_layer_scale*torch.ones(layer_dim_coef[i])))
        self.final=nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim,num_classes),
            nn.Softmax()
            )
        self.apply(self._init_weight)
    def _init_weight(self,m):
        if isinstance(m,(nn.Conv1d,nn.Linear)):
            nn.init.trunc_normal_(m.weight,std=0.02)
    def forward(self,x):


        x=self.small(x.permute(0,2,1).float()).permute(0,2,1)

        in_=[]
        in_.append(x*self.layer_parameter_in[0])
        for i in range(len(self.first_in)):
            x=self.first_in[i](x)

            in_.append(x)

        #print(x.shape)
        
        #x=x.permute(0,2,1)
        in_.pop()
        in_.reverse()
        for i in range(len(self.first_out)):
            x=self.first_out[i](x)
            x=self.first_out[i](x)+in_[i]
        return self.final(x.mean(1))
