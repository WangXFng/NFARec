
import json
from datetime import datetime


# user trajectory/history
user_traj = {}

# to map str to int. For example, "mh_-eMZ6K5RLWhZyISBwA" -> 0
users, items = {}, {}

# to recover int to str. For example,  0 -> "mh_-eMZ6K5RLWhZyISBwA"
int_users_id, int_item_id = {}, {}

with open("../yelp_academic_dataset_review.json", 'r', encoding='utf-8') as f:
    line = f.readline()  # read the file line by line
    count = 0
    while line:
        # print(line)
        count += 1
        if count % 1000000 == 0:
            print(count)
            # break

        review = json.loads(line)   # each line is a json format for a review

        # print(review['business_id'], review['user_id'])
        if review['business_id'] not in items:
            items[review['business_id']] = len(items)  # give an initial increment item id starting from 0
            int_item_id[items[review['business_id']]] = review['business_id']

        if review['user_id'] not in users:
            users[review['user_id']] = len(users) # give an initial increment user id starting from 0
            int_users_id[users[review['user_id']]] = review['user_id']

        user_id, item_id = users[review['user_id']], items[review['business_id']]

        # # if you need timestamp
        # datetime_obj = datetime.strptime(review['date'], "%Y-%m-%d %H:%M:%S")
        # timestamp = datetime.timestamp(datetime_obj)
        if user_id not in user_traj:
            user_traj[user_id] = [item_id]
        else:
            user_traj[user_id].append(item_id)

        # read next line
        line = f.readline()

    f.close()
    print(count)


import numpy as np

# to record how many interactions that users and items have
u = np.zeros(len(users))
b = np.zeros(len(items))

# to record interactions of users and items, it is to solve:
#       once an item is filtered, then the item can not appear in any user's visit history.
#       If the item is removed, a user's interaction count is reduced from 15 to 14, then the user needs to be removed

u_a = [[] for i in range(len(users))]
b_a = [[] for i in range(len(items))]

for user_id in user_traj:
    traj = user_traj[user_id]
    for item_id in traj:
        u[user_id] += 1
        b[item_id] += 1

        u_a[user_id].append(item_id)
        b_a[item_id].append(user_id)

flag = True
while flag:  # if there are users or items with less than 15 interactions, continue
    where_u = np.where((u > 0) & (u < 15))[0]
    flag_u = False
    for u1 in where_u:  # each u1 is the one that needs to be removed
        if len(u_a[u1]) > 0:
            flag_u = True

            for t_b in u_a[u1]:  # the interaction count of a business appearing in u1's history decreases by 1
                b[t_b] -= 1

            u_a[u1] = []  # clean u1's history
    u[where_u] = 0

    where_b = np.where((b > 0) & (b < 15))[0]
    flag_b = False
    for b1 in where_b:  # each b1 is the business that needs to be removed
        if len(b_a[b1]) > 0:
            flag_b = True

            for t_u in b_a[b1]:
                u[t_u] -= 1

            b_a[b1] = []

    b[where_b] = 0

    flag = flag_b or flag_u

# keep users who have more than 0 interactions. "> 14" is also ok
user_indices = np.where(u > 0)[0]
item_indices = np.where(b > 0)[0]


# rerank users and saved in new_user_map, for example:
#               "user_id":"mh_-eMZ6K5RLWhZyISBhwA" -> user_id: 0
#               "user_id":"OyoGAe7OKpv6SyGZT5g77Q" -> user_id: 1
new_user_map, new_item_map = {}, {}
for i, u_id in enumerate(user_indices):
    new_user_map[int_users_id[u_id]] = i
for i, b in enumerate(item_indices):
    new_item_map[int_item_id[b]] = i


import pickle

data_output = open('../new_user_map.pkl', 'wb')
pickle.dump({'new_user_map': new_user_map}, data_output)
data_output.close()
print('File: ', 'new_user_map.pkl', 'saved!')

data_output = open('../new_item_map.pkl', 'wb')
pickle.dump({'new_item_map': new_item_map}, data_output)
data_output.close()
print('File: ', 'new_item_map.pkl', 'saved!')


print(len(new_user_map))
print(len(new_item_map))