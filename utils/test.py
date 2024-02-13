import torch
import numpy as np

# 生成一个5x5的矩阵，元素从1到25
matrix = torch.arange(1, 26).view(5, 5)

print(matrix)

list = torch.tensor(np.array([1, 2, 3]))

sub_adj = torch.zeros((3, 3), device='cuda:0')

sub_adj[:, :] = matrix[list, list]

print(matrix[list, list])

print(sub_adj)


# sub_adj[:, :] = matrix[list, list]

sub_adj2 = torch.zeros((3, 3), device='cuda:0')

sub_adj2[:, :] = matrix[list, :][:, list]

print(sub_adj2)