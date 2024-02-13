# def parse(self, data):
#     user_traj, user_sentiment, user_times = [[[] for j in range(self.user_num)] for i in range(3)]
#     for eachline in data:
#         uid, lid, score, times = eachline.strip().split()
#         uid, lid, score, times = int(uid), int(lid), int(score), int(times)
#         try:
#             user_traj[uid].append(lid + 1)
#             user_times[uid].append(len(user_times[uid]))
#             if times > 3:
#                 user_sentiment[uid].append(2)
#             else:
#                 user_sentiment[uid].append(1)
#         except Exception as e:
#             print(uid, len(user_traj))
#     return user_traj, user_sentiment, user_times

#
training_set = open('../../../Food.com/Food.com_train.txt', 'r').readlines()
tuning_set = open('../../../Food.com/Food.com_tune.txt', 'r').readlines()
# eachline = f.readline()
training_set.extend(tuning_set)
user_training = {}
for eachline in training_set:
    uid, lid, score, times = eachline.strip().split()
    uid, lid, score, times = int(uid), int(lid), int(score), int(times)

    if uid not in user_training:
        user_training[uid] = [(lid, score)]
    else:
        user_training[uid].append((lid, score))

count = 0
# w = open('train.txt', 'w')
for i in range(len(user_training)):
    for j in user_training[i]:
        lid, score = j
        # w.write(str(i) + " " + str(lid) + " " + str(score) + "\n")
        count += 1
# w.close()

#

test_set = open('../../../Food.com/Food.com_test.txt', 'r').readlines()
user_test = {}
for eachline in test_set:
    uid, lid, score, times = eachline.strip().split()
    uid, lid, score, times = int(uid), int(lid), int(score), int(times)

    if uid not in user_test:
        user_test[uid] = [(lid, score)]
    else:
        user_test[uid].append((lid, score))

w = open('test.txt', 'w')
for i in range(len(user_test)):
    for j in user_test[i]:
        lid, score = j
        # w.write(str(i) + " " + str(lid) + " " + str(score) + "\n")
        count += 1
# w.close()

print(count)