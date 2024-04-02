import Constants as C
import numpy as np
import torch.nn.functional as F
import torch
import sys
sys.path.append("..")


def read_interaction():

    start_time = time.time()
    directory_path = './data/{dataset}/'.format(dataset=C.DATASET)

    train_data = open(directory_path + '{dataset}_train.txt'.format(dataset=C.DATASET), 'r').readlines()
    train_data.extend(open(directory_path + '{dataset}_tune.txt'.format(dataset=C.DATASET), 'r').readlines())


    interaction_matrix = torch.zeros((C.USER_NUMBER, C.ITEM_NUMBER), device='cuda:0')
    adjacency_matrix = torch.zeros((C.ITEM_NUMBER, C.ITEM_NUMBER), device='cuda:0')

    print(interaction_matrix.size())
    count = 0
    for eachline in train_data:
        uid, lid, scores, times = eachline.strip().split()
        uid, lid, scores, times = int(uid), int(lid), int(scores), int(times)
        if scores > 3:
            interaction_matrix[uid][lid] = 1
        else:
            interaction_matrix[uid][lid] = -1
        count += 1
        if count % 500000 == 0:
            print(count, time.time()-start_time)

    correlation_matrix = torch.matmul(interaction_matrix.T, interaction_matrix)
    correlation_matrix = correlation_matrix / correlation_matrix.max()

    for i in range(C.USER_NUMBER):
        # poi_rev = interaction_matrix[:, i]
        nwhere = torch.where(interaction_matrix[i]!=0)[0]
        for j in nwhere:
            adjacency_matrix[j][nwhere] = 1

    np.save(directory_path + 'adjacency_matrix.npy', adjacency_matrix.cpu().numpy())
    np.save(directory_path + 'correlation_matrix.npy', correlation_matrix.cpu().numpy())


import time


def main():
    read_interaction()


if __name__ == '__main__':
    main()



