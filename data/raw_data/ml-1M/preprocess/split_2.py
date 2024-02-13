import random
# train = []
# test = []

users = {}

item_max = 0

with open('../ratings.dat') as f:
    for line in f:
        user_id, item_id, score, time = line.strip().split('::')
        # new_line = '\t'.join(items[:-1])+'\n'
        # if int(items[-2])<4:
        #     continue

        user_id, item_id, score, time = int(user_id)-1, int(item_id)-1, int(score), int(time)
        item_max = max(item_id, item_max)
        if user_id in users:
            users[user_id].append(str(item_id)+'\t'+str(score)+'\t'+str(time))
        else:
            users[user_id] = [str(item_id)+'\t'+str(score)+'\t'+str(time)]

    f.close()

print(item_max)

train = open('../../../ml-1M/ml-1M_train.txt', 'w')
tune = open('../../../ml-1M/ml-1M_tune.txt', 'w')
test = open('../../../ml-1M/ml-1M_test.txt', 'w')
for u in users:
    j_ = len(users[u])
    for i, j in enumerate(users[u]):
        # for i, sj in enumerate(j):
        if i < int(j_*0.7):
            train.write(str(u)+"\t"+j+'\n')
        elif i < int(j_*0.8):
            tune.write(str(u)+"\t"+j+'\n')
        else:
            test.write(str(u)+"\t"+j+'\n')

train.close()
tune.close()
test.close()
# with open('train.txt','w') as f:
#     f.writelines(train)
#
# with open('test.txt','w') as f:
#     f.writelines(test)