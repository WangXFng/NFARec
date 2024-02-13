import torch.nn as nn
from utils.Utils import *
import Constants as C
import torch.nn
from utils import Utils
# from gMLP.gmlp import gMLP
#
from model.THP.THP import THPEncoder
from model.HGC.HGC import HGCEncoder
from model.transformer.Layers import EncoderLayer
# from transformerls.lstransformer import TransformerLS


class Encoder(nn.Module):
    def __init__(
            self,
            num_types, d_model, n_layers, n_head, dropout):
        super().__init__()
        self.d_model = d_model

        self.hgc_layers = nn.ModuleList([
            HGCEncoder(d_model, n_head)
            for _ in range(n_layers)])

        if C.ENCODER == 'Transformer':
            self.layer_stack = nn.ModuleList([
                EncoderLayer(d_model, d_model, 1, int(d_model ** 0.5), int(d_model ** 0.5), dropout=dropout, normalize_before=False)
                for _ in range(n_layers)])
        elif C.ENCODER == 'THP':
            self.layer_stack = nn.ModuleList([
                THPEncoder(d_model, n_head)
                for _ in range(n_layers)])
        elif C.ENCODER == 'NHP':
            # # OPTIONAL recurrent layer, this sometimes helps
            self.rnn = RNN_layers(d_model, 128)

        self.linear = nn.Linear(d_model, C.ITEM_NUMBER)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, event_type, enc_output, matrices):
        """ Encode event sequences via masked self-attention. """
        (adj_matrix, cor_matrix) = matrices
        # get individual adj
        local_cor = torch.zeros((event_type.size(0), event_type.size(1), event_type.size(1)), device='cuda:0')
        for i, e in enumerate(event_type):
            # Thanks to Lin Fang for reminding me to correct a mistake here.
            local_cor[i] = adj_matrix[e - 1, :][:, e - 1]
            # performance can be enhanced by adding the element in the diagonal of the normalized adjacency matrix.
            local_cor[i] += adj_matrix[e - 1, e - 1]
        if C.ENCODER == 'Transformer':
            slf_attn_mask_subseq = get_subsequent_mask(event_type)  # M * L * L
            slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)  # M x lq x lk
            slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

            non_pad_mask = get_non_pad_mask(event_type)
            for enc_layer in self.layer_stack:
                enc_output, _ = enc_layer(
                    enc_output,
                    non_pad_mask=non_pad_mask,  # non_pad_mask
                    slf_attn_mask=slf_attn_mask,  # slf_attn_mask
                )
            attn_output = enc_output
            # return _, _.mean(1), correlation_output.mean(1), outputs

        if C.ENCODER == 'THP':
            # non_pad_mask = get_non_pad_mask(event_type)
            # mask = Utils.get_non_pad_mask_for_matrix(event_type)
            # for i, enc_layer in enumerate(self.layer_stack):
            #     attn_output, correlation_output = enc_layer(enc_output, local_cor, event_type)
            for i, enc_layer in enumerate(self.layer_stack):
                attn_output = enc_layer(enc_output, local_cor, event_type)
        elif C.ENCODER == 'NHP':
            try:
                non_pad_mask = get_non_pad_mask(event_type)
                attn_output = self.rnn(enc_output, non_pad_mask)
            except Exception as e:
                attn_output = enc_output

        for i, enc_layer in enumerate(self.hgc_layers):
            correlation_output, outputs = enc_layer(enc_output, local_cor, cor_matrix, event_type)

            # # 32, L, 1024,   32 L N
            #
            # # mask = Utils.get_attn_key_pad_mask(event_type, cor_matrix).transpose(1, 2)
            # # outputs = self.linear(correlation_output.mean(1)) * (cor_matrix[event_type-1].mean(1))
            # # * (cor_matrix[event_type-1] * mask).mean(1)
            #
            # outputs = torch.matmul(correlation_output.transpose(1, 2), cor_matrix[event_type-1]).mean(2)

            #
            # outputs = correlation_output.mean(1)

            # outputs = []
            # for i, (o, e) in enumerate(zip(correlation_output, event_type)):
            #     # o: L, 1024
            #     outputs.append(o * cor_matrix[e-1])
            #     # outputs.append(torch.matmul(o.transpose(0, 1), cor_matrix[e-1]))
            #
            # outputs = torch.stack(outputs, 0)

        return attn_output, attn_output.mean(1), correlation_output.mean(1), outputs
        # return enc_output, enc_output.mean(1), correlation_output.mean(1)
        # return enc_output[:, -1, :]


class RNN_layers(nn.Module):

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types, opt):
        super().__init__()

        self.dim = dim

        self.linear = nn.Linear(dim, C.ITEM_NUMBER)
        nn.init.xavier_uniform_(self.linear.weight)
        # nn.init.uniform_(self.linear.weight)

        self.implicit_conv_features = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.implicit_conv_features.weight)

        if 'delta' in opt: self.delta_3 = opt.delta
        else: self.delta_3 = 1.2
        # self.delta_3 = 1

    def forward(self, user_seq_rep, embeddings, user_gra_rep, user_gra2_rep):
        outputs = []

        out = user_seq_rep.matmul(embeddings.T[:, 1:])
        out = F.normalize(out, p=2, dim=-1, eps=1e-05)
        outputs.append(out)

        if C.DATASET == 'Beauty' or C.DATASET == 'Yelp2023' or C.DATASET == 'Food.com' or C.DATASET == 'Amazon-book':
            out = user_gra_rep.matmul(embeddings.T[:, 1:])
            out = F.normalize(out, p=2, dim=-1, eps=1e-05)
            outputs.append(out)

        out = self.linear(user_gra2_rep)
        out = F.normalize(out, p=2, dim=-1, eps=1e-05)
        outputs.append(out * self.delta_3)

        outputs = torch.stack(outputs, dim=0).sum(0)   # torch.Size([28484, 768])
        out = torch.tanh(outputs)
        return out


class Model(nn.Module):
    def __init__(
            self, num_types, d_model=256, n_layers=4, n_head=4, dropout=0.1, device=0, opt=None):
        super(Model, self).__init__()

        self.event_emb = nn.Embedding(num_types+1, d_model, padding_idx=C.PAD)  # dding 0
        # self.user_emb = nn.Embedding(C.USER_NUMBER, d_model, padding_idx=C.PAD)  # dding 0

        self.encoder = Encoder(
            num_types=num_types, d_model=d_model,
            n_layers=n_layers, n_head=n_head, dropout=dropout)
        self.num_types = num_types

        # self.THPEncoder = THPEncoder(d_model)

        self.predictor = Predictor(d_model, num_types, opt)

        # self.linear = nn.Linear(d_model, d_model)
        # nn.init.xavier_uniform_(self.linear.weight)

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, 3)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # self.dropout = nn.Dropout(0.4)

        # self.norm_emb = nn.Embedding(1, d_model, padding_idx=C.PAD)  # dding 0

    def forward(self, user_id, event_type, matrices):
        mask = Utils.get_non_pad_mask(event_type)
        enc_output = self.event_emb(event_type) * mask

        candidates = self.event_emb.weight
        attn_output, user_seq_rep, user_gra_rep, user_gra2_rep = self.encoder(event_type, enc_output, matrices)

        prediction = self.predictor(user_seq_rep, candidates, user_gra_rep, user_gra2_rep)

        return attn_output, prediction, user_seq_rep, user_gra_rep

    # def transformer_hawkes_process(self, event_type):
    #     mask = Utils.get_non_pad_mask(event_type)
    #     enc_output = self.event_emb(event_type) * mask
    #
    #     # candidates = self.event_emb.weight
    #     attn_output = self.THPEncoder(enc_output, event_type)
    #     # # event_embeddings, enc_output.mean(1), correlation_output.mean(1), outputs
    #     # enc_output, user_embeddings, correlation_output, outputs = self.encoder(event_type, enc_output, matrices)
    #     #
    #     # prediction = self.predictor(enc_output, user_embeddings, candidates, correlation_output, outputs)
    #
    #     return attn_output
