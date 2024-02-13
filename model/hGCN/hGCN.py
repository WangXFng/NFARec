import torch.nn.functional as F
import torch.nn as nn
import Constants as C
from utils import Utils
import torch


class hGCNEncoder(nn.Module):

    def __init__(self, d_model, n_head):
        super().__init__()

        self.heads = nn.ModuleList([
            hGCNLayer(d_model, d_model) for _ in range(n_head)
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

    def forward(self, output, local_cor, event_type):
        slf_attn_mask = None
        if event_type is not None:
            slf_attn_mask_subseq = Utils.get_subsequent_mask(event_type)  # M * L * L
            slf_attn_mask_keypad = Utils.get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)  # M x lq x lk
            slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

            # output = output * self.get_non_pad_mask(event_type)
        outputs, correlation_outputs = [], []
        # correlation_outputs = []

        # latest_k_mask = self.get_latest_k_mask(event_type, 20)
        for head in self.heads:
            output, correlation_output = head(output, local_cor, None, slf_attn_mask)
            outputs.append(output)
            correlation_outputs.append(correlation_output)
        outputs = torch.stack(outputs, dim=0)
        correlation_outputs = torch.stack(correlation_outputs, dim=0)
        return outputs.sum(0), correlation_outputs.sum(0)
        # return correlation_outputs.sum(0)


class hGCNLayer(nn.Module):
    def __init__(self, d_model, d_k):
        super(hGCNLayer, self).__init__()

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

    def forward(self, output, local_cor, latest_k_mask, slf_attn_mask):
        # # #
        # q, k = self.w_qs(output), self.w_ks(output)
        q, k = output, output
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2)) * slf_attn_mask  # * mask
        attn = F.normalize(attn, p=2, dim=-1, eps=1e-05)

        attn_output = torch.matmul(attn, F.elu(self.linear(output)))
        # mask = torch.zeros(sparse_norm_adj.size(), device='cuda:0', dtype=torch.float32)
        # for i, (sub_mask, star) in enumerate(zip(mask, event_star)):
        #     pos_, neg_ = torch.where((star != 0) & (star <= 3))[0], torch.where(star>3)[0]
        #     mask[i][pos_, pos_] = 1
        #     mask[i][neg_, neg_] = 1

        # sparse_norm_adj = F.normalize(local_cor, p=2, dim=-1, eps=1e-05)
        correlation_output = torch.matmul(local_cor, F.elu(self.linear3(output)))

        return attn_output, correlation_output



