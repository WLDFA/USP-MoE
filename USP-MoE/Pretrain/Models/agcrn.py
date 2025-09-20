import torch
import torch.nn as nn
import torch.nn.functional as F

class AGCRN(nn.Module):
    '''
    Reference code: https://github.com/LeiBAI/AGCRN
    '''
    def __init__(self, input_dim = 1,output_dim = 6,horizon = 6,embed_dim = 32, rnn_unit = 64, num_layer = 2, cheb_k = 64):
        super(AGCRN, self).__init__()

        self.encoder = AVWDCRNN(input_dim, rnn_unit, cheb_k, embed_dim, num_layer)
        self.end_conv = nn.Conv2d(1, horizon * output_dim, kernel_size=(1, rnn_unit), bias=True)
        self.node_embed = None

    def forward(self, source,A_hat):  # (b, t, n, f)
        source = source.x
        source = source.permute(0,2,1,3)
        bs, _, node_num, _ = source.shape

        if self.node_embed is None or self.node_embed.shape[0] != node_num:
            self.node_embed = nn.Parameter(torch.randn(node_num, 32), requires_grad=True)

        device = source.device
        self.node_embed.data = self.node_embed.data.to(device)

        init_state = self.encoder.init_hidden(bs, node_num,device)
        output, _ = self.encoder(source, init_state, self.node_embed)
        output = output[:, -1:, :, :]
        pred = self.end_conv(output)
        pred = pred.squeeze(-1).permute(0, 2, 1)
        return pred


class AVWDCRNN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, num_layer):
        super(AVWDCRNN, self).__init__()
        assert num_layer >= 1, 'At least one DCRNN layer in the Encoder.'
        self.num_layer = num_layer
        self.cheb_k = cheb_k
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.embed_dim = embed_dim
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layer):
            self.dcrnn_cells.append(AGCRNCell(dim_out, dim_out, cheb_k, embed_dim))


    def forward(self, x, init_state, node_embed):
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layer):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embed)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden


    def init_hidden(self, batch_size, node_num):
        init_states = []
        for i in range(self.num_layer):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size, node_num))
        return torch.stack(init_states, dim=0)


class AGCRNCell(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)


    def forward(self, x, state, node_embed):
        #state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embed))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embed))
        h = r*state + (1-r)*hc
        return h


    def init_hidden_state(self, batch_size, node_num):
        return torch.zeros(batch_size, node_num, self.hidden_dim)


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))


    def forward(self, x, node_embed):
        node_num = node_embed.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embed, node_embed.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num), supports]

        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embed, self.weights_pool)
        bias = torch.matmul(node_embed, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        return x_gconv