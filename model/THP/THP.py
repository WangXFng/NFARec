import torch.nn.functional as F
import torch.nn as nn
import Constants as C
from utils import Utils
import torch


class THPEncoder(nn.Module):

    def __init__(self, d_model, n_head):
        super().__init__()

        self.heads = nn.ModuleList([
            THPLayer(d_model, d_model) for _ in range(n_head)
        ])

    def get_non_pad_mask(self, seq):
        """ Get the non-padding positions. """
        return seq.ne(C.PAD).type(torch.float).unsqueeze(-1)

    def get_latest_k_mask(self, seq, k):
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
        outputs = []
        for head in self.heads:
            output = head(output, slf_attn_mask)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)
        return outputs.sum(0)


class THPLayer(nn.Module):

    def __init__(self, d_model, d_k):
        super(THPLayer, self).__init__()

        self.linear = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.linear.weight)

        self.temperature = d_model ** 0.5
        self.dropout = nn.Dropout(0.1)

    def forward(self, output, slf_attn_mask):
        q, k = output, output
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2)) * slf_attn_mask  # * mask
        attn = F.normalize(attn, p=2, dim=-1, eps=1e-05)

        attn_output = torch.matmul(attn, F.elu(self.linear(output)))

        return attn_output



