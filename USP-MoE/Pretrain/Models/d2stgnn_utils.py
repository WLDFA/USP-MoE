# model_utils.py
import torch
import meta_d2stgnn  # 替换为实际的模型文件和类名

def create_model(args, data_args, node_num):
    # 创建模型参数字典
    model_args = {
        'num_feat': args.num_feat,
        'num_hidden': args.num_hidden,
        'node_hidden': args.node_hidden,
        'time_emb_dim': args.time_emb_dim,
        'layer': args.layer,
        'k_t': args.k_t,
        'k_s': args.k_s,
        'tpd': args.tpd,
        'gap': args.gap,
        'dy_graph': args.dy_graph,
        'sta_graph': args.sta_graph
    }
    
    # 其他参数
    other_args = {
        'node_num': node_num
    }
    
    # 初始化模型
    model = meta_d2stgnn(model_args, **other_args)
    return model