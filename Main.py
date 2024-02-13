from preprocess.Dataset import Dataset as dataset
from model.Models import Model
from tqdm import tqdm
import torch.optim as optim
import Constants as C
import numpy as np
import argparse
import optuna
import torch
from utils import Utils, metric
import time
import shutil
import os

if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T


def train_epoch(model, user_dl, matrices, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()
    [pre, rec, map_, ndcg] = [[[] for i in range(4)] for j in range(4)]
    for batch in tqdm(user_dl, mininterval=2, desc='  - (Training)   ', leave=False):
        optimizer.zero_grad()

        """ prepare data """
        user_idx, event_type, event_times, sentiment, test_label = map(lambda x: x.to(opt.device), batch)

        """ forward """
        att_output, prediction, user_seq_rep, user_gra_rep = model(user_idx, event_type, matrices)

        """ compute metric """
        metric.pre_rec_top(pre, rec, map_, ndcg, prediction, test_label, event_type)

        """ backward """
        loss = Utils.type_loss(prediction, event_type, test_label, opt)

        if C.ENCODER in {'THP', 'NHP'}:
            event_ll, non_event_ll = Utils.log_likelihood(model, att_output, event_times, sentiment)
            event_loss = -torch.mean(event_ll - non_event_ll)
            if C.DATASET in C.DICT: loss += event_loss / C.DICT[C.DATASET]

        loss.backward(retain_graph=True)
        """ update parameters """
        optimizer.step()

    results_np = map(lambda x: [np.around(np.mean(i), 5) for i in x], [pre, rec, map_, ndcg])
    return results_np


def eval_epoch(model, user_valid_dl, matrices, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()
    user_idx_set = []
    # recom_lists_map = {}
    # user_idx_ndcg = np.zeros(C.USER_NUMBER)
    user_seq_embeddings = torch.zeros((C.USER_NUMBER, opt.d_model), device='cuda:0')
    user_gra_embeddings = torch.zeros((C.USER_NUMBER, opt.d_model), device='cuda:0')

    [pre, rec, map_, ndcg] = [[[] for i in range(4)] for j in range(4)]
    with torch.no_grad():
        for batch in tqdm(user_valid_dl, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare test data """
            user_idx, event_type, event_times, sentiment, test_label = map(lambda x: x.to(opt.device), batch)
            user_idx_set.extend(user_idx.cpu().numpy().tolist())

            """ forward """
            enc_out, prediction, user_seq_rep, user_gra_rep = model(user_idx, event_type, matrices)  # X = (UY+Z) ^ T

            user_seq_embeddings[user_idx] = user_seq_rep
            user_gra_embeddings[user_idx] = user_gra_rep

            """ compute metric """
            metric.pre_rec_top(pre, rec, map_, ndcg, prediction, test_label, event_type)

    results_np = map(lambda x: [np.around(np.mean(i), 5) for i in x], [pre, rec, map_, ndcg])
    return results_np, user_seq_embeddings, user_gra_embeddings


def train(model, data, optimizer, scheduler, opt):
    """ Start training. """

    best_ = [np.zeros(4) for i in range(4)]
    (user_valid_dl, user_dl, adj_matrix, cor_matrix) = data
    matrices = (adj_matrix, cor_matrix)
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i + 1, ']')

        np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
        start = time.time()
        [pre, rec, map_, ndcg] = train_epoch(model, user_dl, matrices,  optimizer, opt)
        # print('\r(Training)  P@k:{pre},    R@k:{rec}, \n'
        #       '(Training)map@k:{map_}, ndcg@k:{ndcg}, '
        #       'elapse:{elapse:3.3f} min'
        #       .format(elapse=(time.time() - start) / 60, pre=pre, rec=rec, map_=map_, ndcg=ndcg))

        start = time.time()
        [pre, rec, map_, ndcg], user_seq_embeddings, user_gra_embeddings = eval_epoch(model, user_valid_dl, matrices, opt)
        print('\r(Test)  P@k:{pre},    R@k:{rec}, \n'
              '(Test)map@k:{map_}, ndcg@k:{ndcg}, '
              'elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60, pre=pre, rec=rec, map_=map_, ndcg=ndcg))
        # print('\r(Test) R@k:{rec}, ndcg@k:{ndcg}, '
        #       'elapse:{elapse:3.3f} min'
        #       .format(elapse=(time.time() - start) / 60, rec=rec, ndcg=ndcg))

        scheduler.step()
        if best_[-1][1] < ndcg[1]: best_ = [pre, rec, map_, ndcg]

        # path = './result/{}/emb/'.format(C.DATASET)
        # folder = "{accuracy:8.5f}/".format(accuracy=best_[-1][1])
        # if not os.path.exists(path) or len(os.listdir(path)) == 0:
        #     if not os.path.exists(path): os.mkdir(path)
        #     if not os.path.exists(path+folder): os.mkdir(path+folder)
        #     # np.save(path+folder+"wo_item.npy", model.event_emb.weight.detach().cpu().numpy())
        #     np.save(path+folder+"user_seq_embeddings.npy", user_seq_embeddings.cpu().numpy())
        #     np.save(path+folder+"user_gra_embeddings.npy", user_gra_embeddings.cpu().numpy())
        #     np.save(path+folder+"item.npy", model.event_emb.weight.detach().cpu().numpy())
        # for file_name in os.listdir(path):
        #     if float(file_name) < best_[-1][1]:
        #         shutil.rmtree(path+file_name)
        #         os.mkdir(path + folder)
        #         # np.save(path+folder+"wo_item.npy", model.event_emb.weight.detach().cpu().numpy())
        #         np.save(path+folder+"user_seq_embeddings.npy", user_seq_embeddings.cpu().numpy())
        #         np.save(path+folder+"user_gra_embeddings.npy", user_gra_embeddings.cpu().numpy())
        #         np.save(path+folder+"item.npy", model.event_emb.weight.detach().cpu().numpy())

    return best_[-1][1]


def main(trial):
    """ Main function. """
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.device = torch.device('cuda')

    # # # optuna setting for tuning hyperparameters
    # opt.n_layers = trial.suggest_int('n_layers', 2, 2)
    # opt.d_inner_hid = trial.suggest_int('n_hidden', 512, 1024, 128)
    # opt.d_k = trial.suggest_int('d_k', 512, 1024, 128)
    # opt.d_v = trial.suggest_int('d_v', 512, 1024, 128)
    # opt.n_head = trial.suggest_int('n_head', 1, 5, 1)
    # # opt.d_rnn = trial.suggest_int('d_rnn', 128, 512, 128)
    # opt.d_model = trial.suggest_int('d_model', 128, 1024, 128)
    # opt.dropout = trial.suggest_uniform('dropout_rate', 0.5, 0.7)
    # opt.smooth = trial.suggest_uniform('smooth', 1e-2, 1e-1)
    # opt.lr = trial.suggest_uniform('learning_rate', 0.00008, 0.0002)

    lambda_delta = {
        'ml-1M': [1.47, 3.99],
        'Beauty': [0.6168, 3.7435],
        'Yelp2023': [0.12079, 1.48861],
        'Food.com': [0.1152865, 3.81303],
        'Amazon-book': [0.24558344, 3.53373],
    }

    if C.DATASET in lambda_delta:
        [opt.beta, opt.lambda_] = lambda_delta[C.DATASET]
    else:
        # opt.lambda_, opt.delta = trial.suggest_uniform('lambda', 0.1, 4), trial.suggest_uniform('delta', 0.1, 4)
        opt.beta, opt.lambda_ = 0.5, 1

    opt.lr = 0.01
    opt.epoch = 20
    opt.n_layers = 1  # 2
    opt.batch_size = 16
    opt.dropout = 0.5
    opt.smooth = 0.03

    if C.DATASET == 'ml-1M': opt.d_model, opt.n_head, opt.epoch, opt.batch_size = 1024, 1, 30, 32
    elif C.DATASET == 'Yelp2023': opt.d_model, opt.n_head, opt.epoch, opt.batch_size = 1024, 3, 15, 16
    elif C.DATASET == 'Food.com': opt.d_model, opt.n_head, opt.epoch, opt.batch_size = 1024, 1, 6, 32
    elif C.DATASET == 'Beauty': opt.d_model, opt.n_head, opt.epoch, opt.batch_size = 1024, 1, 16, 32
    elif C.DATASET == 'Amazon-book': opt.d_model, opt.n_head, opt.epoch, opt.batch_size = 1024, 2, 16, 32
    else: opt.d_model, opt.n_head = 1024, 1

    print('[Info] parameters: {}'.format(opt))
    num_types = C.ITEM_NUMBER
    num_user = C.USER_NUMBER

    """ prepare model """
    model = Model(
        num_types=num_types,
        d_model=opt.d_model,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        device=opt.device,
        opt=opt
    )
    model = model.cuda()

    """ loading data"""
    ds = dataset()
    print('[Info] Loading data...')
    user_dl = ds.get_user_dl(opt.batch_size)
    user_valid_dl = ds.get_user_valid_dl(opt.batch_size)
    adj_matrix = ds.ui_adj
    cor_matrix = ds.cor_mat

    data = (user_valid_dl, user_dl, adj_matrix, cor_matrix)
    """ optimizer and scheduler """
    parameters = [
                  {'params': model.parameters(), 'lr': opt.lr},
                  ]
    optimizer = torch.optim.Adam(parameters)  # , weight_decay=0.01
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    return train(model, data, optimizer, scheduler, opt)


if __name__ == '__main__':
    # main()
    study = optuna.create_study(direction="maximize")
    study.optimize(main, n_trials=100)

    # df = study.trials_dataframe()
    #
    # print("Best trial:")
    # trial = study.best_trial
    # print("  Value: ", trial.value)
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))

