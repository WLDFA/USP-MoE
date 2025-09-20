
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import pdb

class LSTM(nn.Module):
    def __init__(self,input_dim = 1, init_dim = 32, hid_dim = 64, end_dim=32,output_dim = 6, layer = 2, dropout = 0.2):
        super(LSTM, self).__init__()   
        self.start_conv = nn.Conv2d(in_channels=input_dim, 
                                    out_channels=init_dim, 
                                    kernel_size=(1,1))

        self.lstm = nn.LSTM(input_size=init_dim, hidden_size=hid_dim, num_layers=layer, batch_first=True, dropout=dropout)
        
        self.end_linear1 = nn.Linear(hid_dim, end_dim)
        self.end_linear2 = nn.Linear(end_dim, output_dim)


    def forward(self, input,A_hat):  # (b, t, n, f)
        # 确保数据张量的维度为 (batch_size, num_nodes, history_length, 1)
        x = input.x  # 假设数据张量存储在 input 中

        # 将形状从 (batch_size, num_nodes, history_length, 1) 转换为 (batch_size, 1, num_nodes, history_length)
        x = x.permute(0,3,1,2).float()
        # 通过卷积层处理
        x = self.start_conv(x)
        
        # 调整形状为 (batch_size * num_nodes, history_length, init_dim)
        batch_size, _, num_nodes, history_length = x.shape
        x = x.reshape(batch_size * num_nodes, history_length, -1)
           
        # 通过 LSTM 层处理
        out, _ = self.lstm(x)
        
        # 取 LSTM 的最后一个时间步的输出
        x = out[:, -1, :]
        
        # 通过线性层处理
        x = F.relu(self.end_linear1(x))
        x = self.end_linear2(x)
        
        # 将输出重新调整为 (batch_size, num_nodes, output_dim)
        x = x.reshape(batch_size, num_nodes, -1)
        
        return x