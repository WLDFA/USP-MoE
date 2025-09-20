import torch
import torch.nn as nn
import math
from einops import rearrange
import sys

from TimeTransformer.encoder import Encoder, CrossAttention_Encoder, AdaIN_Encoder
from TimeTransformer.decoder import Decoder
from TimeTransformer.utils import generate_original_PE, generate_regular_PE
import TimeTransformer.causal_convolution_layer as causal_convolution_layer
import torch.nn.functional as F
import pdb

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

# 添加MoE相关组件

class Expert(nn.Module):
    """单个专家FFN层"""
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class SparseRouter(nn.Module):
    """稀疏门控网络（Router）"""
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
    def forward(self, x: torch.Tensor) -> tuple:
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # 计算门控分数
        gate_scores = self.gate(x_flat)  # [batch_size * seq_len, num_experts]
        
        # 选择top_k个专家
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_scores, dim=-1)
        
        return top_k_probs, top_k_indices

class MoELayer(nn.Module):
    """混合专家层"""
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 创建专家网络
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        
        # 门控网络
        self.router = SparseRouter(d_model, num_experts, top_k)
        
        # 负载均衡损失权重
        self.load_balance_weight = 0.01
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # 获取路由信息
        top_k_probs, top_k_indices = self.router(x)
        
        # 初始化输出
        output = torch.zeros_like(x_flat)
        
        # 为每个token分配到对应的专家
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]  # 第i个最佳专家的索引
            expert_probs = top_k_probs[:, i].unsqueeze(-1)  # 对应概率
            
            # 为每个专家计算输出
            for expert_idx in range(self.num_experts):
                mask = (expert_indices == expert_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    output[mask] += expert_probs[mask] * expert_output
        
        return output.view(batch_size, seq_len, d_model)

class MoEEncoder(nn.Module):
    """集成MoE的编码器层"""
    def __init__(self, d_model: int, nhead: int, num_experts: int, top_k: int = 2, 
                 d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.moe = MoELayer(d_model, num_experts, top_k, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # 自注意力层
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # MoE层替代传统FFN
        moe_output = self.moe(x)
        x = self.norm2(x + self.dropout(moe_output))
        
        return x
    def freeze_base_model(self):
        """第一阶段：冻结基础模型参数，只训练专家"""
        for name, param in self.named_parameters():
            if 'moe.experts' not in name:
                param.requires_grad = False
    
    def unfreeze_all(self):
        """第二阶段：解冻所有参数进行端到端训练"""
        for param in self.parameters():
            param.requires_grad = True
    
    def load_pretrained_experts(self, expert_paths: list):
        """加载预训练的专家模型"""
        for i, (layer_idx, expert_idx, path) in enumerate(expert_paths):
            if layer_idx < len(self.layers_encoding):
                layer = self.layers_encoding[layer_idx]
                if hasattr(layer, 'moe') and expert_idx < len(layer.moe.experts):
                    expert_state = torch.load(path)
                    layer.moe.experts[expert_idx].load_state_dict(expert_state)
                    print(f"Loaded expert {expert_idx} for layer {layer_idx} from {path}")

class TransformerMoE(nn.Module):
    '''
    Transformer with Mixture of Experts (MoE)
    '''

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_kgEmb: int,
                 d_timeEmb: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 spatialloc: list,
                 num_experts: int = 8,  # 专家数量
                 top_k: int = 2,        # 每次激活的专家数
                 attention_size: int = None,
                 layernum: int = 0,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 ifkg: bool = True,
                 pe_period: int = None,
                 learned_sinusoidal_cond: bool = False,   
                 random_fourier_features: bool = False,
                 learned_sinusoidal_dim: int = 16,
                 use_moe_layers: list = None,  # 指定哪些层使用MoE，如[0,2,4]                 
                 ):
        """Create transformer structure with MoE layers."""
        super().__init__()

        self._d_model = d_model
        self.kgEmb_dim = d_kgEmb
        self.timeEmb_dim = d_timeEmb
        self.channels = d_input
        self.ifkg = ifkg
        self.spatialloc = spatialloc
        step_dim = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.layernum = layernum
        self.self_condition = False
        
        # 确定哪些层使用MoE
        self.use_moe_layers = use_moe_layers or list(range(N))

        # 混合使用传统Encoder和MoE Encoder
        self.layers_encoding = nn.ModuleList()
        for i in range(N):
            if i in self.use_moe_layers:
                # 使用MoE编码器
                self.layers_encoding.append(
                    MoEEncoder(d_model, h, num_experts, top_k, dropout=dropout)
                )
            else:
                # 使用传统编码器
                self.layers_encoding.append(
                    Encoder(d_model, q, v, h, 
                           attention_size=attention_size,
                           dropout=dropout,
                           chunk_mode=chunk_mode)
                )

        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        
        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        # 位置编码设置
        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        self.kgEmb_mlp = nn.Sequential(
            nn.Linear(self.kgEmb_dim, d_model),
            nn.ReLU(),  
            nn.Linear(d_model, d_model)  
        )
        
        self.timeEmb_mlp = nn.Sequential(
            nn.Linear(self.timeEmb_dim, d_model),
            nn.ReLU(),  
            nn.Linear(d_model, d_model)  
        )
        
        self.timelinear = nn.Linear(self.timeEmb_dim, d_model)
        
        self.last_mlp = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(d_model, d_model)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, kgEmb: torch.Tensor, 
                timeEmb: torch.Tensor, x_self_cond: bool) -> torch.Tensor:

        x2 = x.permute(0,2,1) 

        # 时间嵌入
        timeEmb = self.timelinear(timeEmb)
        timeEmb = timeEmb.unsqueeze(1)
        timeEmb = torch.repeat_interleave(timeEmb, self.layernum, dim=1)

        # 步长嵌入
        step = self.step_mlp(t) 
        step = step.unsqueeze(1)  
        step_emb = torch.repeat_interleave(step, self.layernum, dim=1)

        # 嵌入模块
        encoding = self._embedding(x2)
        encoding.add_(step_emb)
        encoding.add_(timeEmb)
        
        # 知识图谱嵌入
        if self.ifkg:
            kgEmb = self.kgEmb_mlp(kgEmb)
            kgEmb = kgEmb.unsqueeze(1)        
            kgEmb = torch.repeat_interleave(kgEmb, 160, dim=1)
            encoding[:, self.spatialloc[0]:self.spatialloc[1], :].add_(kgEmb)
        
        K = self.layernum

        # 位置编码
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # 编码器堆栈 - 混合MoE和传统层
        for i, layer in enumerate(self.layers_encoding):
            if i in self.use_moe_layers:
                # MoE层
                encoding = layer(encoding)
            else:
                # 传统编码器层
                encoding = layer(encoding)
        
        output = self._linear(encoding)
        return output.permute(0,2,1)

    def get_expert_utilization(self) -> dict:
        """获取专家使用率统计"""
        utilization = {}
        for i, layer in enumerate(self.layers_encoding):
            if hasattr(layer, 'moe'):
                # 这里可以添加专家使用率统计逻辑
                utilization[f'layer_{i}'] = "MoE layer statistics"
        return utilization
    


