import torch.nn.functional as F
import torch.nn as nn
import Constants as C
import torch


class HGCEncoder(nn.Module):

    def __init__(self, d_model, n_head):
        super().__init__()

        self.heads = nn.ModuleList([
            HGCLayer(d_model, d_model) for _ in range(n_head)
        ])

    def get_non_pad_mask(self, seq):
        """ Get the non-padding positions. """
        return seq.ne(C.PAD).type(torch.float).unsqueeze(-1)


    def get_latest_k_mask(self, seq, k):
        # 生成一个对角线上前k个元素为1的矩阵，其余为0
        bz, seq_len = seq.size()
        mask = torch.zeros(seq_len, seq_len, device='cuda:0')
        # mask = torch.zeros_like(seq)
        for i in range(seq_len):
            start = max(i, i - k + 1)
            end = min(seq_len, i + k)
            mask[i, start:end] = 1

        return mask.unsqueeze(0)

    def forward(self, output, local_cor, cor_matrix, event_type):
        outputs, correlation_outputs = [], []
        for head in self.heads:
            output, correlation_output = head(output, local_cor, cor_matrix, event_type)
            outputs.append(output)
            correlation_outputs.append(correlation_output)
        outputs = torch.stack(outputs, dim=0)
        correlation_outputs = torch.stack(correlation_outputs, dim=0)
        return outputs.sum(0), correlation_outputs.sum(0)
        # return correlation_outputs.sum(0)


class HGCLayer(nn.Module):
    def __init__(self, d_model, d_k):
        super(HGCLayer, self).__init__()

        self.linear = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.linear.weight)
        #
        # self.linear2 = nn.Linear(d_model, d_model)
        # nn.init.xavier_uniform_(self.linear2.weight)

        self.linear3 = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.linear3.weight)

        # self.w_qs = nn.Linear(d_model, d_k, bias=False)
        # self.w_ks = nn.Linear(d_model, d_k, bias=False)
        # # self.w_vs = nn.Linear(d_model, n_head, bias=False)
        # nn.init.xavier_uniform_(self.w_qs.weight)
        # nn.init.xavier_uniform_(self.w_ks.weight)
        # # nn.init.xavier_uniform_(self.w_vs.weight)
        #
        # # self.eps = 1
        self.temperature = d_model ** 0.5
        self.dropout = nn.Dropout(0.1)

    def forward(self, output, local_cor, cor_matrix, event_type):

        correlation_output = torch.matmul(local_cor, F.elu(self.linear3(output)))

        outputs = torch.matmul(correlation_output.transpose(1, 2), cor_matrix[event_type - 1]).mean(2)

        return correlation_output, outputs



