from PredictionModel.NetSet import StgnnSet
import sys
from PredictionModel.datasets import *
from PredictionModel.utils import *
from tqdm import tqdm
import pdb

#返回指定节点在新索引列表中的位置
def getNewIndex(nodeindex, addr):
    A = np.load(addr)
    nodeset = []
    for i in range(A.shape[0]):
        if A[i][nodeindex] != 0  or  A[nodeindex][i] != 0:
            nodeset.append(i)
    # nodeset = nodeset2
    for i in range(len(nodeset)):
        if nodeset[i]==nodeindex:
            return i


def  evalParams(params, config, device, logger, targetDataset, writer, epoch, basemodel):
    
    data_args, task_args, model_args = config['data'], config['task'], config['model']
    
    #pdb.set_trace()
    model = StgnnSet(data_args, task_args, model_args, basemodel).to(device=device)
    
    node_num = 0
    epochs = 0
    if targetDataset == 'None':
        datasetsall = ['DC','BM','man']
    elif targetDataset == 'TrafficNone':
        datasetsall = ['Orange_small', 'Riverside_small', 'San_Bernardino_small', 'Ventura']
    else:
        datasetsall = [targetDataset]
        
    results = []
    for dataset in datasetsall:
        if dataset == 'DC':
            node_num = 194
            locbias = 0
            
        elif dataset == 'Orange_small':
            node_num = 239
            locbias = 0
        elif dataset == 'shenzhen_4':
            node_num = 4
            locbias =  0
        elif dataset == 'Orange_STGCN':
            node_num = 953
            locbias = 0
        elif dataset == 'Orange_LSTM':
            node_num = 953
            locbias = 0
        elif dataset == 'Placer':
            node_num = 69
            locbias = 0
        elif dataset == 'ventura':
            node_num = 131
            locbias = 0
        elif dataset == 'Riverside':
            node_num = 482
            locbias = 0
        elif dataset == 'Marin_2':
            node_num = 140
            locbias = 0
        elif dataset == 'Placer_2':
            node_num = 69
            locbias = 0
        elif dataset == 'Riverside_2':
            node_num = 482
            locbias = 0
        elif dataset == 'ca_pre':
            node_num = 600
            locbias = 8000 if targetDataset=='TrafficNone' else 0
        elif dataset == 'ca_pre100':
            node_num = 100
            locbias = 8500 if targetDataset=='TrafficNone' else 0
        elif dataset == 'ca_pre300':
            node_num = 300
            locbias = 8300 if targetDataset=='TrafficNone' else 0

        step = 0
        params = params.reshape((params.shape[0],-1))
        #pdb.set_trace()
        for node_index in tqdm(range(node_num)):
            model = StgnnSet(data_args, task_args, model_args, model=basemodel).to(device=device)
            
            test_dataset = traffic_dataset2(data_args, task_args, node_index, "test", False, dataset)
            test_meanstd = [test_dataset.mean, test_dataset.std]
            test_dataloader = DataLoader(test_dataset, batch_size=task_args['test_batch_size'], shuffle=False, num_workers=0, pin_memory=False, drop_last=True)
            AAddr = data_args[dataset]['adjacency_matrix_path']
            
            newindex = getNewIndex(node_index,AAddr)
            #pdb.set_trace()
            param = params[node_index + locbias]
            outputs, y_label = model.task4eval(param, newindex, node_index, test_dataloader, logger, test_meanstd, basemodel) 
            

            np.save('/home/wll/G/GPD/result/{}/{}_predict.npy'.format(dataset,node_index), outputs)
            np.save('/home/wll/G/GPD/result/{}/{}_truth.npy'.format(dataset,node_index), y_label)

            if step == 0 :
                out = outputs
                truth = y_label
                step = 1
            else:
                out = np.concatenate((out, outputs),axis=2)
                truth = np.concatenate((truth, y_label),axis=2)
        #pdb.set_trace()
        result = metric_func(pred=out, y=truth, times=task_args['pred_num'])
        results.append(result)

        if dataset == 'DC':
            writer.add_scalar('DC_mae', np.mean(result['MAE']), epoch)
            writer.add_scalar('DC_rmse', np.mean(result['RMSE']), epoch)
        elif dataset == 'BM':
            writer.add_scalar('BM_mae', np.mean(result['MAE']), epoch)
            writer.add_scalar('BM_rmse', np.mean(result['RMSE']), epoch)
        elif dataset == 'man':
            writer.add_scalar('man_mae', np.mean(result['MAE']), epoch)
            writer.add_scalar('man_rmse', np.mean(result['RMSE']), epoch)

    metricsum = []
    for result in results:
        metricsum.append(result_print(result, logger,info_name='Evaluate'))
        
    return sum(metricsum)
        
        
        
        