import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
from torch.autograd import Variable
import sys
import yaml
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch_geometric.nn import GATConv
from torchsummary import summary
import pdb

#数据和邻接矩阵相乘
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))#矩阵乘法
        return x.contiguous()#返回连续的内存空间
#线性变换层
class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=False)

    def forward(self,x):
        return self.mlp(x)
#图卷积层
class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in#（2*3+1）*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            # print("a shape is", a.shape)
            # print("x shape is", x.shape)
            # a shape is torch.Size([627])
            # x shape is torch.Size([64, 32, 627, 12])
            # 'ncvl,vw->ncwl'
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)#正则化
        return h


class v_GWN(nn.Module):
    def __init__(self, dropout=0.2, gcn_bool=True, in_dim=1,out_dim=6,
        residual_channels=32,dilation_channels=32,skip_channels=32,
        end_channels=32,kernel_size=2,blocks=4,layers=2):
        #初始化网络层
        '''
        
        dropout=0.2, gcn_bool=True, in_dim=2,out_dim=6,
        residual_channels=32,dilation_channels=32,skip_channels=32,
        end_channels=32,kernel_size=2,blocks=4,layers=2
        
        dropout=0.2, gcn_bool=True, in_dim=2,out_dim=6,
        residual_channels=16,dilation_channels=8,skip_channels=16,
        end_channels=16,kernel_size=2,blocks=4,layers=2
        
        dropout=0.3, gcn_bool=True, in_dim=2,out_dim=6,
        residual_channels=32,dilation_channels=32,skip_channels=256,
        end_channels=512,kernel_size=2,blocks=4,layers=2
        '''
        
        super(v_GWN, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        #存储信息
        self.filter_convs = nn.ModuleList()#过滤卷积层
        self.gate_convs = nn.ModuleList()#门控卷积曾
        self.residual_convs = nn.ModuleList()#残差卷积层
        self.skip_convs = nn.ModuleList()#跳跃连接卷积层
        self.bn = nn.ModuleList()#归一化层
        self.gconv = nn.ModuleList()#图卷积层
        #第一层卷积层
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1), bias=False)#二维卷积层
        receptive_field = 1#感受野

        # All supports are double transition
        self.supports_len = 2

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions膨胀卷积
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation,
                                                   bias=False))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation,
                                                 bias=False))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1), bias=False))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1), bias=False))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=False)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=False)

        self.receptive_field = receptive_field

    def forward(self, input, supports):
        #pdb.set_trace()
        input = input.x
        input = input.permute(0,3,1,2).float()

        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else: 
            x = input
        #pdb.set_trace()
        x = self.start_conv(x)
        skip = 0       
        
        # WaveNet layers
        for i in range(self.blocks * self.layers):#8

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # (dilation, init_dilation) = self.dilations[i]

            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # print(x.shape)
            # dilated convolution
            # print("Conv2d input shape is ", residual.shape)
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            # print("Conv1d input shape is ", residual.shape)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and supports is not None:
                x = self.gconv[i](x,supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)
        #pdb.set_trace()
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = (x.squeeze(-1)).permute(0,2,1)
        return x