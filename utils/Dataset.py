import os
import torch
import numpy as np
import Constants as C
import scipy.sparse as sp
from utils.cal_pairwise import read_interaction

if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T


class Dataset(object):
    def __init__(self):
        self.user_num = C.USER_NUMBER
        self.poi_num = C.ITEM_NUMBER
        self.directory_path = './data/{dataset}/'.format(dataset=C.DATASET)

        self.training_user, self.training_sentiment, self.training_times = self.read_training_data()
        self.tuning_user, self.tuning_sentiment, self.tuning_times = self.read_tuning_data()
        self.test_user, self.test_sentiment, self.test_times = self.read_test_data()

        self.user_data, self.user_valid = self.read_data()
        self.ui_adj = self.load_adjacent_matrix()
        self.cor_mat = self.load_correlation_matrix()

    def parse(self, data):
        user_traj, user_sentiment, user_times = [[[] for j in range(self.user_num)] for i in range(3)]
        for eachline in data:
            uid, lid, score, times = eachline.strip().split()
            uid, lid, score, times = int(uid), int(lid), int(score), int(times)
            try:
                user_traj[uid].append(lid + 1)
                user_times[uid].append(len(user_times[uid]))
                if times > 3:
                    user_sentiment[uid].append(2)
                else:
                    user_sentiment[uid].append(1)
            except Exception as e:
                print(uid, len(user_traj))
        return user_traj, user_sentiment, user_times

    def read_data(self):
        user_data, user_valid = [], []

        for i in range(self.user_num):

            valid_label = self.training_user[i].copy()
            valid_label.extend(self.tuning_user[i])

            valid_times = self.training_times[i].copy()
            valid_times.extend(self.tuning_times[i])

            valid_sentiment = self.training_sentiment[i].copy()
            valid_sentiment.extend(self.tuning_sentiment[i])

            # for training
            # user_data.append((i, self.training_user[i], self.training_sentiment[i], self.tuning_user[i], valid_label, valid_times, ), )
            user_data.append((i, self.training_user[i], self.training_times[i], self.training_sentiment[i], self.tuning_user[i], ), )

            # test_label = valid_label.copy()
            # test_label.extend(self.test_user[i])
            #
            # test_times = valid_times.copy()
            # test_times.extend(self.test_times[i])

            # for testing
            # user_valid.append((i, valid_label, valid_sentiment, self.test_user[i], self.test_times[i], test_label, test_times, ), )
            user_valid.append((i, valid_label, valid_times, valid_sentiment, self.test_user[i], ), )

        return user_data, user_valid

    def read_training_data(self):
        train_file = '{dataset}_train.txt'.format(dataset=C.DATASET)
        return self.parse(open(self.directory_path + train_file, 'r').readlines())

    def read_tuning_data(self):
        tune_file = '{dataset}_tune.txt'.format(dataset=C.DATASET)
        return self.parse(open(self.directory_path + tune_file, 'r').readlines())

    def read_test_data(self):
        test_file = '{dataset}_test.txt'.format(dataset=C.DATASET)
        return self.parse(open(self.directory_path + test_file, 'r').readlines())

    def load_adjacent_matrix(self):
        directory_path = './data/{dataset}/'.format(dataset=C.DATASET)
        train_file = 'adjacency_matrix.npy'
        if not os.path.exists(directory_path + train_file):
            print('adjacency_matrix is not found, generating ...')
            read_interaction()
        print('Loading ', directory_path + train_file, '...')
        ui_adj = np.load(directory_path + train_file)
        ui_adj = sp.csr_matrix(ui_adj)
        print('Computing adj matrix ...')
        ui_adj = torch.tensor(self.normalize_graph_mat(ui_adj).toarray(), device='cuda:0', dtype=torch.float32)
        return ui_adj

    def load_correlation_matrix(self):
        directory_path = './data/{dataset}/'.format(dataset=C.DATASET)
        train_file = 'correlation_matrix.npy'
        if not os.path.exists(directory_path + train_file):
            print('correlation_matrix is not found, generating ...')
            read_interaction()
        print('Loading ', directory_path + train_file, '...')
        cor_mat = np.load(directory_path + train_file)
        cor_mat = sp.csr_matrix(cor_mat)
        cor_mat = torch.tensor(cor_mat.toarray(), device='cuda:0', dtype=torch.float32)

        for i in range(C.NUM_LAYERS-1):
            cor_mat = torch.matmul(cor_mat, cor_mat.T) / cor_mat.size(0) + cor_mat

        cor_mat[cor_mat <= 0] = 1e-9
        # cor_mat[cor_mat != 0] = np.log(cor_mat[cor_mat != 0]+2)
        # cor_mat = torch.tensor(self.normalize_graph_mat(cor_mat).toarray(), device='cuda:0')
        return cor_mat

    def normalize_graph_mat(self, adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        # rowsum[rowsum != 0] = np.exp(rowsum[rowsum != 0]/2)
        # print(rowsum[rowsum != 0].min())
        rowsum[rowsum != 0] = np.log(rowsum[rowsum != 0]+2)
        rowsum[rowsum <= 0] = 1e-9
        # print(rowsum.max(), rowsum.min())
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    # def get_in_degree(self):
    #     in_degree = np.zeros(C.ITEM_NUMBER + 1)
    #     directory_path = './data/{dataset}/'.format(dataset=C.DATASET)
    #     if os.path.exists(directory_path + 'in_degree.npy'):
    #         in_degree = np.load(directory_path + 'in_degree.npy')
    #     else:
    #         all_train_data = open(directory_path + '{dataset}_train.txt'.format(dataset=C.DATASET), 'r').readlines()
    #         all_tune_data = open(directory_path + '{dataset}_tune.txt'.format(dataset=C.DATASET), 'r').readlines()
    #         all_train_data.extend(all_tune_data)
    #         for eachline in all_train_data:
    #             uid, lid, times = eachline.strip().split()
    #             uid, lid, times = int(uid), int(lid), int(times)
    #             in_degree[lid + 1] += 1
    #         np.save(directory_path + 'in_degree.npy', in_degree)
    #     return torch.tensor(in_degree, device='cuda:0', dtype=torch.float16)

    def paddingLong2D(self, insts):
        """ Pad the instance to the max seq length in batch. """
        # max_len = 700
        max_len = max(len(inst) for inst in insts)
        batch_seq = np.array([
            inst[:max_len] + [C.PAD] * (max_len - len(inst))
            for inst in insts])
        return torch.tensor(batch_seq, dtype=torch.long)

    def padding2D(self, insts):
        # max_len = 700
        max_len = max(len(inst) for inst in insts)
        batch_seq = np.array([
            inst[:max_len] + [C.PAD] * (max_len - len(inst))
            for inst in insts])
        return torch.tensor(batch_seq, dtype=torch.float32)

    def user_fn(self, insts):
        """ Collate function, as required by PyTorch. """
        (useridx, event_type, event_time, sentiment, test_label) = list(zip(*insts))
        useridx = torch.tensor(useridx, device='cuda:0')
        event_type = self.paddingLong2D(event_type)
        sentiment = self.paddingLong2D(sentiment)
        event_time = self.paddingLong2D(event_time)
        test_label = self.paddingLong2D(test_label)
        # sequential_event = self.paddingLong2D(sequential_event)
        # sequential_times = self.paddingLong2D(sequential_times)
        return useridx, event_type, event_time, sentiment, test_label  # , sequential_event, sequential_times

    def get_user_dl(self, batch_size):
        user_dl = torch.utils.data.DataLoader(
            self.user_data,
            num_workers=0,
            batch_size=batch_size,
            collate_fn=self.user_fn,
            shuffle=True
        )
        return user_dl

    def get_user_valid_dl(self, batch_size):
        user_valid_dl = torch.utils.data.DataLoader(
            self.user_valid,
            num_workers=0,
            batch_size=batch_size,
            collate_fn=self.user_fn,
            shuffle=True
        )
        return user_valid_dl

