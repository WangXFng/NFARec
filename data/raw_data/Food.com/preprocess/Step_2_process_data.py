
import os

import pickle
import json
from datetime import datetime

# if os.path.exists('user_data.pkl'):
# 1 {'AKJHHD5VEH7VG': 0}

# print(len(new_user_map), new_user_map)
# print(len(new_item_map), new_item_map)

with open('../new_user_map.pkl', 'rb') as f:
    new_user_map = pickle.load(f, encoding='latin-1')['new_user_map']
    f.close()

with open('../new_item_map.pkl', 'rb') as f:
    new_item_map = pickle.load(f, encoding='latin-1')['new_item_map']
    f.close()


print(len(new_user_map))
print(len(new_item_map))

user_traj = {}


with open('../RAW_interactions.csv', encoding='utf-8') as f:
    line = f.readline()
    line = f.readline()
    count = 0
    while line:
        # review = json.loads(line)

        s = line.strip().split(',')
        if len(s) < 5:
            line = f.readline()
            count += 1
            if count % 100000 == 0:
                print(count)
            continue
        user_id, item_id, date, rating, review = s[0], s[1], s[2], s[3], str(s[4:])
        
        try:
            if user_id not in new_user_map or item_id not in new_item_map:
                count += 1
                if count % 1000000 == 0:
                    print(count)
                line = f.readline()
                continue
            int_user_id, int_item_id = new_user_map[user_id], new_item_map[item_id]
    
            # 将时间字符串转换为 datetime 对象
            datetime_obj = datetime.strptime(date, "%Y-%m-%d")
            # 获取时间戳（秒级）
            timestamp = datetime.timestamp(datetime_obj)
    
            if int_user_id not in user_traj:
                user_traj[int_user_id] = [(int_item_id, int(float(rating)), int(timestamp))]
            else:
                user_traj[int_user_id].append((int_item_id, int(float(rating)), int(timestamp),))
                
        except Exception as e:
            # print(line)
            pass
        count += 1
        if count % 1000000 == 0:
            print(count)
        line = f.readline()
    f.close()

train = open('../../../Food.com/Food.com_train.txt', 'w')
tune = open('../../../Food.com/Food.com_tune.txt', 'w')
test = open('../../../Food.com/Food.com_test.txt', 'w')
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