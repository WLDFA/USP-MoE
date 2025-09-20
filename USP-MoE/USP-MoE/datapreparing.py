import sys

import numpy as np
import torch
from numpy import random
from sklearn.datasets import make_moons
import torch
import torch.nn as nn
import pdb


def datapreparing(targetDataset, basemodel):
    if targetDataset=='Marin':
        trainid = list(range(953, 8600))
        genid = list(range(0, 953))
        rawdata = np.load('../Data/1/Marin/modelparams.npy')
        kgEmb = np.load('../Data/1/Marin/spatialprompt.npy')
        timeEmb = np.load('../Data/1/Marin/timeprompt.npy')
    elif targetDataset=='Placer':
        trainid = list(range(953, 8600))
        genid = list(range(0, 953))
        rawdata = np.load('../Data/2/Placer/modelparams.npy')
        kgEmb = np.load('../Data/2/Placer/spatialprompt.npy')
        timeEmb = np.load('../Data/2/Placer/timeprompt.npy')


    genTarget = rawdata[genid]
    training_seq = rawdata[trainid]#642*11888
    print('training_seq', training_seq.shape)
    if basemodel=='v_GWN':
        channel =  32 # turn params to 1D sequence, the channel depends on the model dim 换成gla数据记得改回来 16
    elif basemodel=='v_STGCN5':
        channel = 64
    else:
        channel = 2
    repeatNum = 1 
    #pdb.set_trace()
    training_seq = training_seq.reshape(training_seq.shape[0],channel,-1) #750*32*2871
    genTarget = genTarget.reshape(genTarget.shape[0],channel,-1) #131*32*2871
    training_seq = np.repeat(training_seq, repeatNum, axis=0) #22500*32*2871

    scale = np.max(np.abs(training_seq))  
    print('larger than 1', np.sum(np.abs(rawdata)>1))

    if scale < 1:
        scale = 1
    # scale = 1
    training_seq = training_seq/scale  
    training_seq = training_seq.astype(np.float32)
    genTarget = genTarget/scale 
    genTarget = genTarget.astype(np.float32)


    #处理空间提示和时间提示
    kgEmb = kgEmb.astype(np.float32)

    kgtrainEmb = kgEmb[trainid]#750*128
    kggenEmb = kgEmb[genid]#131*128
    kgtrainEmb= np.repeat(kgtrainEmb, repeatNum, axis=0)   #22500*128
    
        
    timeEmb = timeEmb.astype(np.float32)
    timetrainEmb = timeEmb[trainid]
    timegenEmb = timeEmb[genid]
    timetrainEmb = np.repeat(timetrainEmb, repeatNum, axis=0)
    
    return training_seq, scale, kgtrainEmb, kggenEmb, timetrainEmb, timegenEmb, genTarget
    
    
