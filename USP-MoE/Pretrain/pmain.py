import argparse
import sys
import time
import pdb

import setproctitle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from datasets import *
from datasets import traffic_dataset2
#from Models.MetaKnowledgeLearner import *
from NetSet import *
from torch.utils.data.sampler import *
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils import *


#重新映射节点索引或者在处理子图时，找到原始图中某个节点在子图中的新索引
def getNewIndex(nodeindex, addr):
    A = np.load(addr)
    nodeset = []
    for i in range(A.shape[0]):
        if A[i][nodeindex] != 0  or  A[nodeindex][i] != 0:
            nodeset.append(i)
    for i in range(len(nodeset)):
        if nodeset[i]==nodeindex:
            return i
#数据集的一次完整遍历
def train_epoch(train_dataloader):
    optimizer = optim.Adam(model.parameters(), lr=0.00001)  #初始化优化器，学习率设置为0.001
    train_losses = []
    for step, (data, A_wave) in enumerate(train_dataloader):
        model.train()#训练模式
        optimizer.zero_grad()#梯度清零
        A_wave = A_wave.to(device=args.device)
        A_wave = A_wave.float()
        data = data.to(device=args.device)
        out, meta_graph = model(data, A_wave)#调用模型
        loss_predict = loss_criterion(out, data.y)#计算预测输出与真实标签之间的损失
        loss_reconsturct = loss_criterion(meta_graph, A_wave)#计算重构输出与真实邻接矩阵之间的损失
        loss = loss_predict + loss_reconsturct#总损失
        loss.backward()#计算模型参数的梯度
        optimizer.step()#根据梯度更新模型参数
        # print("loss_predict: {}, loss_reconsturct: {}".format(loss_predict.detach().cpu().numpy(), loss_reconsturct.detach().cpu().numpy()))
        train_losses.append(loss.detach().cpu().numpy())
    return sum(train_losses)/len(train_losses)

def test_epoch(test_dataloader):
    with torch.no_grad():#不计算梯度
        model.eval()#评估模式
        for step, (data, A_wave) in enumerate(test_dataloader):
            A_wave = A_wave.to(device=args.device)
            data = data.to(device=args.device)#数据和邻接矩阵转移到GPU
            out, _ = model(data, A_wave)#调用模型
            if step == 0:
                outputs = out
                y_label = data.y
            else:
                outputs = torch.cat((outputs, out))#拼接输出
                y_label = torch.cat((y_label, data.y))
        outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()#转移到CPU并转换维度
        y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
    return outputs, y_label


parser = argparse.ArgumentParser(description='MAML-based')#创建解析器
parser.add_argument('--config_filename', default='config.yaml', type=str,
                        help='Configuration filename for restoring the model.')
parser.add_argument('--test_dataset', default='gla', type=str)
parser.add_argument('--meta_dim', default=32, type=int)
parser.add_argument('--target_days', default=15, type=int)
parser.add_argument('--model', default='v_GWN', type=str)
parser.add_argument('--loss_lambda', default=1.5, type=float)
parser.add_argument('--memo', default='revise', type=str)
parser.add_argument('--epochs', default=100, type=int)   
parser.add_argument('--taskmode', default='task4', type = str)
parser.add_argument('--nodeindex', default=0, type = int)
parser.add_argument('--iftest', default=True, type = bool)
parser.add_argument("--ifchosenode", action="store_true")
parser.add_argument('--logindex', default='0', type = str)
parser.add_argument('--ifspatial',default=1, type = int)  
parser.add_argument('--ifnewname',default=0, type = int)
parser.add_argument('--aftername',default='', type = str) 
parser.add_argument('--datanum',default=0.7, type = float)  

args = parser.parse_args()#解析参数

print(time.strftime('%Y-%m-%d %H:%M:%S'), "meta_dim = ", args.meta_dim,"target_days = ", args.target_days)



if __name__ == '__main__':  

    logger, filename = setup_logger(args.taskmode, args.test_dataset, args.logindex, args.model, args.aftername)
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    ifWarp = ""
    if os.path.getsize(filename) != 0:        
        ifWarp = "\n\n"
    logger.info(ifWarp + str(current_time) + ": start training")
    logger.info("target dataset: %s" % args.test_dataset)
    # logger.info(parser.parse_args())
    logger.info("model:"+args.model)
    logger.info("taskmode:"+args.taskmode)
    logger.info("ifchosenode:"+str(args.ifchosenode))
    logger.info("logindex:"+str(args.logindex))


    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        logger.info("INFO: GPU")
    else:
        args.device = torch.device('cpu')
        logger.info("INFO: CPU")

    with open(args.config_filename) as f:
        config = yaml.full_load(f)#读入配置文件

    # torch.manual_seed(10)

    data_args, task_args, model_args = config['data'], config['task'], config['model']
    
    model_args['meta_dim'] = args.meta_dim
    model_args['loss_lambda'] = args.loss_lambda
    
    logger.info("batchsize: "+str(task_args['batch_size']))
    logger.info("lr: "+str(model_args['meta_lr']))

    loss_criterion = nn.MSELoss()#计算MSE

    source_training_losses, target_training_losses = [], []
    best_result = ''
    min_MAE = 10000000


    if args.taskmode == 'task4':
        '''
        task4: pretrain node-level model预训练节点级模型
        '''
        node_num = 0
        epochs = 0
        if args.test_dataset == 'bj':
            node_num = 661 # 1010
            epochs = args.epochs
        elif args.test_dataset == 'Marin' or args.test_dataset == 'Placer' :
            node_num = data_args[args.test_dataset]['node_num']
            epochs = args.epochs

        step = 0
        print('{} totalnum: {}'.format(args.test_dataset, node_num))
        for node_index in range(node_num):
            #图神经网络模型实例化
            model = StgnnSet(data_args, task_args, model_args,model=args.model).to(device=args.device)
            

            logger.info("train node_index: {}".format(node_index))
            
            train_dataset = traffic_dataset2(data_args, task_args, node_index, "singlePretrain", args.ifchosenode, test_data=args.test_dataset, target_days=args.target_days, ifspatial = args.ifspatial, datanum = args.datanum)
            #target / singlePretrain
            train_meanstd = [train_dataset.mean,train_dataset.std]
            #数据加载器
            train_dataloader = DataLoader(train_dataset, batch_size=task_args['batch_size'], shuffle=False, num_workers=0, pin_memory=False, drop_last=True)
            test_dataset = traffic_dataset2(data_args, task_args, node_index, "test", args.ifchosenode, test_data=args.test_dataset, ifspatial = args.ifspatial)
            
            test_meanstd = [test_dataset.mean, test_dataset.std]

            test_dataloader = DataLoader(test_dataset, batch_size=task_args['test_batch_size'], shuffle=False, num_workers=0, pin_memory=False, drop_last=True)
            #pdb.set_trace()
            if args.test_dataset=='BM' or  args.test_dataset=='DC' or  args.test_dataset=='man' or  args.test_dataset=='bj':
                if  args.ifspatial == 1:
                    AAddr = data_args[args.test_dataset]['adjacency_matrix_path']
                else:
                    AAddr = data_args[args.test_dataset]['nonspatial_adjacency_matrix_path']
            else:
                AAddr = data_args[args.test_dataset]['adjacency_matrix_path'] if  args.ifchosenode==True else data_args[args.test_dataset]['adjacency_matrix_path']
            newindex = getNewIndex(node_index,AAddr)#获取新索引
            # print("newindex: "+ str(newindex))
            #训练模型
            outputs, y_label = model.node_taskTrain(args.model, newindex, node_index, 
                                                    train_dataloader, test_dataloader,
                                                    train_meanstd, test_meanstd,
                                                    epochs, logger, args.test_dataset, 
                                                    args.ifnewname, args.aftername)  
            target_path = '/home/wll/G/Pretrain/Output/task4_{}_{}'.format(args.test_dataset, args.model)
            if not os.path.exists(target_path):
                os.makedirs(target_path)

            np.save(os.path.join(target_path, '{}_predict.npy'.format(node_index)), outputs)
            np.save(os.path.join(target_path, '{}_label.npy'.format(node_index)), y_label)
            if step == 0 :
                out = outputs
                truth = y_label
                step = 1
            else:
                out = np.concatenate((out, outputs),axis=2)
                truth = np.concatenate((truth, y_label),axis=2)
        
        logger.info("######################################")
        logger.info("###########  final result  ###########")
        result = metric_func(pred=out, y=truth, times=task_args['pred_num'])
        result_print(result, logger,info_name='Evaluate')#输出MSE、RMSE、MAE


    elif args.taskmode == 'task7':
        '''
        task7: after diffusion sample, finetune
        '''
        node_num = 0
        epochs = 0
        if args.test_dataset == 'Marin':
            node_num = 661 # 1010
            epochs = args.epochs
        elif args.test_dataset == 'Placer':
            node_num = 140
            epochs = args.epochs
            params = np.load('../Output/sampleSeq_RealParams_1001.npy')
            print(params.shape)


        step = 0
        # node_set = [626]  
        print('totalnum ',node_num)
        params = params.reshape((params.shape[0],-1))#将二维数组转换为一维数组
        #tqdm库提供执行的进度条
        for node_index in tqdm(range(node_num)):
            # node_index = 5
            #图神经网络模型实例化
            model = StgnnSet(data_args, task_args, model_args, model=args.model).to(device=args.device)

            logger.info("train node_index: {}".format(node_index))
            

            train_dataset = traffic_dataset2(data_args, task_args, node_index, "target", args.ifchosenode, test_data=args.test_dataset, target_days=args.target_days, ifspatial = args.ifspatial, datanum = args.datanum)
            
            train_meanstd = [train_dataset.mean,train_dataset.std]
            
            train_dataloader = DataLoader(train_dataset, batch_size=task_args['batch_size'], shuffle=False, num_workers=0, pin_memory=False, drop_last=True)
            test_dataset = traffic_dataset2(data_args, task_args, node_index, "test", args.ifchosenode, test_data=args.test_dataset, ifspatial = args.ifspatial)
            
            test_meanstd = [test_dataset.mean, test_dataset.std]

            test_dataloader = DataLoader(test_dataset, batch_size=task_args['test_batch_size'], shuffle=False, num_workers=0, pin_memory=False, drop_last=True)
            if args.test_dataset=='BM' or  args.test_dataset=='DC' or  args.test_dataset=='man' or  args.test_dataset=='bj':
                if  args.ifspatial == 1:
                    AAddr = data_args[args.test_dataset]['adjacency_matrix_path']
                else:
                    AAddr = data_args[args.test_dataset]['nonspatial_adjacency_matrix_path']
            else:
                AAddr = data_args[args.test_dataset]['adjacency_matrix_path'] if  args.ifchosenode==True else data_args[args.test_dataset]['adjacency_matrix_path']
            newindex = getNewIndex(node_index,AAddr)
            # print("newindex: "+ str(newindex))
            param = params[node_index]
            #训练模型区别于预训练模型，这里的参数是从扩散采样中得到的
            outputs, y_label = model.node_taskTrain2(param, args.model, newindex, 
                                                    node_index, 
                                                    train_dataloader, test_dataloader,
                                                    train_meanstd, test_meanstd,
                                                    epochs, logger, args.test_dataset, 
                                                    args.ifnewname, args.aftername)  
            
            target_path = '/home/wll/G/Pretrain/Output/task7_{}'.format(args.test_dataset)
            if not os.path.exists(target_path):
                os.makedirs(target_path)

            np.save(os.path.join(target_path, '{}_predict.npy'.format(node_index)), outputs)
            np.save(os.path.join(target_path, '{}_label.npy'.format(node_index)), y_label)
            
            if step == 0 :
                out = outputs
                truth = y_label
                step = 1
            else:
                out = np.concatenate((out, outputs),axis=2)
                truth = np.concatenate((truth, y_label),axis=2)
        
        logger.info("######################################")
        logger.info("###########  final result  ###########")
        result = metric_func(pred=out, y=truth, times=task_args['pred_num'])
        result_print(result, logger,info_name='Evaluate')
        
        
