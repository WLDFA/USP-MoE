'''
Extract parameters from the trained model
'''
import sys

sys.path.append('../../Pretrain')

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from datasets import *
#from Models.MetaKnowledgeLearner import *
from NetSet import *
from torch.utils.data.sampler import *
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils import *

if __name__ == '__main__':

    with open('../config.yaml') as f:
        config = yaml.full_load(f)
    data_args, task_args, model_args = config['data'], config['task'], config['model']
    model = StgnnSet(data_args, task_args, model_args, model='v_GWN')  # v_STGCN5
    ifFirstRegion = True
    # for dataset in ['DC','BM','man']:
    #补充其他数据集
    # for dataset in ['DC','BM','man','metr-la']:
    for dataset in ['ca']:
        if dataset == 'ca':
            nodenum = 8600
        # if dataset == 'Riverside_small':
        #     nodenum = 241 
        # elif dataset == 'San_Bernardino_small':
        #     nodenum = 270 
        # elif dataset == 'Ventura':
        #     nodenum = 131 
            # 为新数据集设置节点数
        for nodeindex in tqdm(range(nodenum)):
            
            lengthlist = []
            startlist  = [0]
            shapelist  = []
            #model.model = torch.load('../Param/Task4/{}_v_GWN_TrafficData/task4_{}.pt'.format('metr-la', nodeindex), map_location=torch.device('cpu'))
            model.model = torch.load('../Param/Task4/{}_v_STGCN5_TrafficData/task4_{}.pt'.format(dataset, nodeindex), map_location=torch.device('cpu'))



            allparams = list(model.model.named_parameters())

            iffirst = True
            for singleparams in allparams:
                astensor = singleparams[1].clone().detach() 
                shapelist.append(astensor.shape)
                tensor1D = astensor.flatten()#展平一维向量
                lengthlist.append(tensor1D.shape[0])
                tensor1D = tensor1D.unsqueeze(0)#增加一个维度
                if iffirst == True:
                    finaltensor = tensor1D
                    iffirst = False
                else:
                    finaltensor = torch.cat((finaltensor,tensor1D), dim = 1)#拼接
                startlist.append(finaltensor.shape[1])
            if ifFirstRegion==True:
                allRegionTensor = finaltensor
                ifFirstRegion = False
            else: 
                allRegionTensor = torch.cat((allRegionTensor, finaltensor), dim=0)#拼接不同区域的参数
    

    np.save('ModelParams_STGCN_16576_8600_ca', allRegionTensor.cpu())  
    print(allRegionTensor.shape)
