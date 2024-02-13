import random
# train = []
# test = []

import numpy as np

users = {}
items = {}
user_traj = {}


int_users_id = {}
int_item_id = {}

filtering_length = 25


with open('ratings_Books.csv', encoding='utf-8') as f:
    line = f.readline()
    # line = f.readline()
    count = 0
    while line:
        user_id, item_id, rating, date = line.strip().split(',')
        try:
            if user_id not in users:
                users[user_id] = len(users)
                int_users_id[users[user_id]] = user_id
            if item_id not in items:
                items[item_id] = len(items)
                int_item_id[items[item_id]] = item_id

            user_id, item_id, score, time = users[user_id], items[item_id], int(float(rating)), date
            if user_id in user_traj:
                user_traj[user_id].append((item_id, score, time,))
            else:
                user_traj[user_id] = [(item_id, score, time,)]
        except Exception as e:
            # print(line)
            pass

        count += 1
        if count % 1000000 == 0:
            print(count)
        line = f.readline()
    f.close()


u = np.zeros(len(users))
b = np.zeros(len(items))

u_a = [[] for i in range(len(users))]
b_a = [[] for i in range(len(items))]

for user_id in user_traj:
    traj = user_traj[user_id]
    for (item_id, score, time) in traj:
        u[user_id] += 1
        b[item_id] += 1

        u_a[user_id].append(item_id)
        b_a[item_id].append(user_id)

flag = True
while flag:
    where_u = np.where((u > 0) & (u < filtering_length))[0]
    flag_u = False
    for u1 in where_u:
        if len(u_a[u1]) > 0:
            # print(u1, u[u1], u_a[u1], b)
            flag_u = True

            for t_b in u_a[u1]:
                b[t_b] -= 1

            u_a[u1] = []
            # print(u1, u[u1], u_a[u1], b)
    u[where_u] = 0

    where_b = np.where((b > 0) & (b < filtering_length))[0]
    flag_b = False
    for b1 in where_b:
        if len(b_a[b1]) > 0:
            flag_b = True

            for t_u in b_a[b1]:
                u[t_u] -= 1

            b_a[b1] = []

    b[where_b] = 0

    flag = flag_b or flag_u

user_indices = np.where(u > 0)[0]
item_indices = np.where(b > 0)[0]

# int_users_id = {}
# int_item_id = {}

# print(int_users_id)
# print(int_item_id)

new_user_map, new_item_map = {}, {}
for i, u_id in enumerate(user_indices):
    # print(u_id)
    # print(u[u_id])
    # print(int_users_id[u_id])
    # print(u_a[u_id])
    # for j in u_a[u_id]:
    #     print(b[j])
    new_user_map[int_users_id[u_id]] = i
    # break
for i, b in enumerate(item_indices):
    new_item_map[int_item_id[b]] = i


# print()

print(len(new_user_map), new_user_map)
print(len(new_item_map), new_item_map)

import pickle

data_output = open('new_user_map.pkl', 'wb')
pickle.dump({'new_user_map': new_user_map}, data_output)
data_output.close()
print('File: ', 'new_user_map.pkl', 'saved!')

data_output = open('new_item_map.pkl', 'wb')
pickle.dump({'new_item_map': new_item_map}, data_output)
data_output.close()
print('File: ', 'new_item_map.pkl', 'saved!')

# print(len(items))
#
# train = open('Beauty_train.txt','w')
# tune = open('Beauty_tune.txt','w')
# test = open('Beauty_test.txt','w')
# for u in user_traj:
#     j_ = len(user_traj[u])
#     for i, j in enumerate(user_traj[u]):
#         # for i, sj in enumerate(j):
#         if i < int(j_*0.7):
#             train.write(str(u)+"\t"+j+'\n')
#         elif i < int(j_*0.8):
#             tune.write(str(u)+"\t"+j+'\n')
#         else:
#             test.write(str(u)+"\t"+j+'\n')
#
# train.close()
# tune.close()
# test.close()
# # with open('train.txt','w') as f:
# #     f.writelines(train)
# #
# # with open('test.txt','w') as f:
# #     f.writelines(test)