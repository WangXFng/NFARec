
import os

import pickle

# if os.path.exists('user_data.pkl'):
# 1 {'AKJHHD5VEH7VG': 0}

# print(len(new_user_map), new_user_map)
# print(len(new_item_map), new_item_map)

with open('new_user_map.pkl', 'rb') as f:
    new_user_map = pickle.load(f, encoding='latin-1')['new_user_map']
    f.close()

with open('new_item_map.pkl', 'rb') as f:
    new_item_map = pickle.load(f, encoding='latin-1')['new_item_map']
    f.close()


print(len(new_user_map))
print(len(new_item_map))

user_traj = {}


with open('ratings_Beauty.csv') as f:
    line = f.readline()
    line = f.readline()
    count = 0
    while line:
        # print(line)
        user_id, item_id, score, times = line.strip().split(',')
        score, times = int(float(score)), int(times)

        # if user_id == 'AKJHHD5VEH7VG':
        #     print(item_id)
        #     print((item_id in new_item_map))
        #     if item_id in new_item_map:
        #         print(new_user_map[user_id], new_item_map[item_id])

        if user_id not in new_user_map or item_id not in new_item_map:
            count += 1
            line = f.readline()
            continue

        int_user_id = new_user_map[user_id]
        int_item_id = new_item_map[item_id]

        if int_user_id not in user_traj:
            user_traj[int_user_id] = [(int_item_id, score, times)]
        else:
            user_traj[int_user_id].append((int_item_id, score, times,))

        count += 1
        if count % 100000 == 0:
            print(count)
        line = f.readline()
    f.close()

train = open('../../../Beauty/Beauty_train.txt', 'w')
tune = open('../../../Beauty/Beauty_tune.txt', 'w')
test = open('../../../Beauty/Beauty_test.txt', 'w')
for u in range(len(user_traj)):
    j_ = len(user_traj[u])
    for i, j in enumerate(user_traj[u]):
        # for i, sj in enumerate(j):
        (int_item_id, score, times) = j
        if i < int(j_*0.7):
            train.write(str(u)+"\t"+str(int_item_id)+"\t"+str(score)+"\t"+str(times)+'\n')
        elif i < int(j_*0.8):
            tune.write(str(u)+"\t"+str(int_item_id)+"\t"+str(score)+"\t"+str(times)+'\n')
        else:
            test.write(str(u)+"\t"+str(int_item_id)+"\t"+str(score)+"\t"+str(times)+'\n')