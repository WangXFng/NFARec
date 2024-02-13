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


# def sentiment_loss(sentiment_out, model, event_type, event_time, test_label, test_times, opt):
#     sentiment_out = torch.squeeze(sentiment_out[:, :], -1)
#     # sentiment_out 32, 100, 1
#     positive_hots = torch.zeros(sentiment_out.size(0), sentiment_out.size(1), device='cuda:0', dtype=torch.float32)
#     negative_hots = torch.zeros(sentiment_out.size(0), sentiment_out.size(1), device='cuda:0', dtype=torch.float32)
#
#     concated_type = torch.concat([event_time, test_times], -1)
#     positive_hots[concated_type==1], negative_hots[concated_type==-1] = 1, -1
#
#     # for i, (e, et, tl, tt) in enumerate(zip(event_type, event_time, test_label, test_times)):
#     #     positive_hots[i][et==1], positive_hots[i][tt==1] = 1, 1
#     #     negative_hots[i][et==-1], positive_hots[i][tt==-1] = -1, -1
#     #     # positive_hots[i][et[et!=0]-1], positive_hots[i][tt[tt!=0]-1] = opt.beta, opt.lambda_
#     #
#     # # for i, (et, tt) in enumerate(zip(event_time, test_times)):
#     # #     positive_hots[i][et[et!=0]-1], positive_hots[i][tt[tt!=0]-1] = opt.beta, opt.lambda_
#
#     pos_log_prb, neg_log_pro = F.logsigmoid(sentiment_out), F.logsigmoid(1 - sentiment_out)
#     # multi_hots = multi_hots * (1 - opt.smooth) + (1 - multi_hots) * opt.smooth / C.ITEM_NUMBER
#     predict_loss = -(positive_hots * pos_log_prb + (-negative_hots) * neg_log_pro)
#
#     loss = torch.sum(predict_loss)
#
#     return loss


# def mmd_loss(event_type, test_label, users_embeddings, model):
#     mmd_loss = []
#     for e, l, ue in zip(event_type, test_label, users_embeddings):
#         e, l = e[e != 0], l[l != 0]
#         mmd_loss.append(mmd_rbf(model.event_emb(torch.concat([e, l], dim=-1)), ue.unsqueeze(0)))
#     return sum(mmd_loss) / len(mmd_loss)


# def bpr_loss(prediction, user_emb, event_type, test_label, model, in_degree):
#     losses = []
#     top_n = 100
#     target_ = torch.ones(event_type.size()[0], C.ITEM_NUMBER, device='cuda:0', dtype=torch.double)
#     for i, (e, l) in enumerate(zip(event_type, test_label)):
#         e, l = e[e != 0]-1, l[l != 0]-1
#         target_[i][e] = 0
#         target_[i][l] = 0
#     prediction = prediction * target_
#
#     top_ = torch.topk(prediction, top_n, -1, sorted=True)[1]
#     # for top, l in zip(top_, label):
#     for ue, e, l, top in zip(user_emb, event_type, test_label, top_):
#         # e, l = e[e != 0], l[l != 0]
#         # pos_ = torch.concat([e, l], dim=-1)
#         # print(ue.unsqueeze(0).size(), (model.event_emb(pos_)).size())
#         # print(in_degree[pos_-1].size())
#         # pos_score = (torch.mul(ue.unsqueeze(0), model.event_emb(pos_))*in_degree[pos_-1].unsqueeze(-1)).sum()
#         neg_score = (torch.mul(ue.unsqueeze(0), model.event_emb(top))*in_degree[top].unsqueeze(-1)).sum()
#         # loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
#         loss = -torch.log(10e-8 + torch.sigmoid( - neg_score))
#         losses.append(torch.mean(loss))
#     return sum(losses)/len(losses)
#
# import torch


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
        将源域数据和目标域数据转化为核矩阵，即上文中的K
        Params:
            source: 源域数据（n * len(x))
            target: 目标域数据（m * len(y))
            kernel_mul:
            kernel_num: 取不同高斯核的数量
            fix_sigma: 不同高斯核的sigma值
        Return:
            sum(kernel_val): 多个核矩阵之和
    '''
    # print(source.size(), target.size())
    n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)
#
#
def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
        计算源域数据和目标域数据的MMD距离
        Params:
            source: 源域数据（n * len(x))
            target: 目标域数据（m * len(y))
            kernel_mul:
            kernel_num: 取不同高斯核的数量
            fix_sigma: 不同高斯核的sigma值
        Return:
            loss: MMD loss
    '''
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算
#
#
# def InfoNCE(view1, view2, temperature):
#     view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
#     pos_score = (view1 * view2).sum(dim=-1)
#     pos_score = torch.exp(pos_score / temperature)
#     ttl_score = torch.matmul(view1, view2.transpose(0, 1))
#     ttl_score = torch.exp(ttl_score / temperature).sum(dim=1) + 1e-8
#     cl_loss = torch.log(pos_score / ttl_score)
#     return torch.mean(cl_loss)


# def ImbalancedInfoNCE(view1, view2, temperature):
#     view1, view2 = F.normalize(view1.unsqueeze(1), dim=1), F.normalize(view2.unsqueeze(0), dim=1)
#     pos_score = (view1 * view2).mean(dim=-1)
#     pos_score = torch.exp(pos_score / temperature)
#     ttl_score = torch.matmul(view1, view2.transpose(1, 2))
#     ttl_score = torch.exp(ttl_score / temperature).mean(dim=1)
#     cl_loss = -torch.log(pos_score / ttl_score)
#     return torch.mean(cl_loss)


# def cal_cl_loss(prediction, event_type, test_label, users_embeddings, model):
#     losses = []
#     top_n = 100
#     target_ = torch.ones(event_type.size()[0], C.ITEM_NUMBER, device='cuda:0', dtype=torch.double)
#     for i, (e, l) in enumerate(zip(event_type, test_label)):
#         e, l = e[e != 0]-1, l[l != 0]-1
#         target_[i][e] = 0
#         target_[i][l] = 0
#
#     top_ = torch.topk(prediction * target_, top_n, -1, sorted=True)[1]
#
#     # for top, l in zip(top_, label):
#     for ue, e, l, top in zip(users_embeddings, event_type, test_label, top_):
#         e, l = e[e != 0], l[l != 0]
#         top = top [:len(e)]
#         # cl_loss = InfoNCE(ue.unsqueeze(0), model.event_emb(torch.concat([e, l], dim=-1)), 0.2)
#         # cl_loss = InfoNCE(ue.unsqueeze(0), model.event_emb(e), 0.2)
#         poi_embeddings = model.event_emb(e)
#         # random_noise = torch.rand_like(poi_embeddings).cuda()
#         # noised_poi_embeddings = poi_embeddings + torch.sign(poi_embeddings) * F.normalize(random_noise, dim=-1) * 0.2
#         cl_loss = InfoNCE(poi_embeddings[:min(len(e), 100)], model.event_emb(top), 0.4)
#         losses.append(cl_loss)
#     # item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)
#     return sum(losses)/len(losses)


# def mmd_loss(event_type, test_label, users_embeddings, model):
#     mmd_loss = []
#     for e, l, ue in zip(event_type, test_label, users_embeddings):
#         e, l = e[e != 0], l[l != 0]
#         mmd_loss.append(mmd_rbf(model.event_emb(e), ue.unsqueeze(0)))
#     return sum(mmd_loss) / len(mmd_loss) # / lambda_
#
#
# def uniformity(x, t=2):
#     x = F.normalize(x, dim=-1)
#     return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def kl_div(P, Q):
    divergence = F.kl_div(Q.softmax(-1).log(), P.softmax(-1), reduction='sum')
    return divergence


def l2_reg_loss(reg, model, event_type, in_degree, user_embeddings):
    emb_loss = 0
    # kl_loss = 0
    for u, e in zip(user_embeddings, event_type):
        e = e[e != 0]
        r = model.event_emb(e)
        emb_loss += torch.norm(r, p=2).sum() * reg

        # kl_loss += mmd_rbf(u.unsqueeze(0), r) * reg

        # sorted_id = torch.sort(in_degree[e])[1]
        # len_ = len(sorted_id)//4
        # l = model.event_emb(e[sorted_id[:len_]])
        # r = model.event_emb(e[sorted_id[-len_:]])
        # kl_loss += mmd_rbf(r, l) * reg * 2

        # r = model.event_emb(t)
        # emb_loss += torch.norm(r, p=2).sum() * (reg * 0.7)

        # id = (1-0.5/(in_degree[e])) * reg  # ** 0.5
        # emb_loss += (torch.norm(r * id.unsqueeze(-1), p=2)).sum()

        # if Constants.DATASET in {'Yelp2018', "douban-book"}:
        #     id = (1-0.5/(in_degree[e])) * reg  # ** 0.5
        #     emb_loss += (torch.norm(r * id.unsqueeze(-1), p=2)).sum()
        # else:
        #     emb_loss += torch.norm(r, p=2).sum() * reg

        # emb_loss += torch.norm(user_output, p=2).sum()  # * reg
        # emb_loss += uniformity(r)/100
    # emb_loss += (torch.norm(model.event_emb.weight, dim=-1, p=2) * model.event_emb.weight.mean(-1).unsqueeze(-1)).sum() * reg
    return emb_loss / len(event_type)  # + kl_loss/len(event_type)


# def popularity_loss(popularity_pred, in_degree):
#     return (popularity_pred - in_degree) ** 2

