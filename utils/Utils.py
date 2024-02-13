import torch.nn.functional as F

import Constants
import Constants as C
import torch

import math


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """
    assert seq.dim() == 2
    return seq.ne(C.PAD).type(torch.float).unsqueeze(-1)


def get_non_pad_mask_for_matrix(seq):
    mask = torch.matmul(seq.ne(C.PAD).type(torch.float).unsqueeze(2), seq.ne(C.PAD).type(torch.float).unsqueeze(1))
    return mask!=0


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(C.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device)
    temp_time /= (time[:, :-1] + 1).unsqueeze(2)
    # data.matmul(model.event_emb.weight.T[:, 1:])[:, 1:, :]  # model.linear(data)[:, 1:, :]
    temp_hid = model.linear(data)[:, 1:, :]  # data.matmul(model.event_emb.weight.T[:, 1:])[:, 1:, :]
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


def log_likelihood(model, data, time, types):
    """ Log-likelihood of sequence. """

    non_pad_mask = get_non_pad_mask(types).squeeze(2)

    type_mask = torch.zeros([*types.size(), 3], device=data.device)
    for i in range(3):
        type_mask[:, :, i] = (types == i + 1).bool().to(data.device)

    all_hid = model.linear(data)  # data.matmul(model.event_emb.weight.T[:, 1:])  # model.linear(data)
    all_lambda = softplus(all_hid, model.beta)
    type_lambda = torch.sum(all_lambda * type_mask, dim=2)

    # event log-likelihood
    event_ll = compute_event(type_lambda, non_pad_mask)
    event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased(model, data, time, non_pad_mask, type_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll


def type_loss(prediction, label, test_label, opt):
    """ Event prediction loss, cross entropy or label smoothing. """
    prediction = torch.squeeze(prediction[:, :], 1)

    multi_hots = torch.zeros(label.size(0), C.ITEM_NUMBER, device='cuda:0', dtype=torch.float32)
    for i, (t, tl) in enumerate(zip(label, test_label)):
        multi_hots[i][t[t!=0]-1], multi_hots[i][tl[tl!=0]-1] = opt.beta, opt.lambda_

    log_prb = F.logsigmoid(prediction)
    multi_hots = multi_hots * (1 - opt.smooth) + (1 - multi_hots) * opt.smooth / C.ITEM_NUMBER
    predict_loss = -(multi_hots * log_prb)

    loss = torch.sum(predict_loss)

    return loss

