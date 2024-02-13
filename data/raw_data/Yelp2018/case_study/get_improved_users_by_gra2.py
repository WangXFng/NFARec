import numpy as np
import pickle

# f = open('full_user_idx_ndcg.npy', 'r')

full_user_idx_ndcg = np.load('../full_user_idx_ndcg.npy')

# user_idx_ndcg = np.load('user_idx_ndcg_wo_gra2.npy')
#
# index_ = np.where((full_user_idx_ndcg>user_idx_ndcg) & (user_idx_ndcg==0))[0]
#
# print(index_)
# print(len(index_))
#
with open('../new_user_map.pkl', 'rb') as f:
    new_user_map = pickle.load(f, encoding='latin-1')['new_user_map']
    f.close()
#
# int_2_id = {}
#
# for u in new_user_map:
#     int_2_id[new_user_map[u]] = u
# #
# # # print(len(new_user_map))
# #
# user_idx_str = {}
# import json
#
# for i in index_:
#     user_idx_str[int_2_id[i]] = 1
#
#
# user_reviews = {}
#
# with open("yelp_academic_dataset_review.json", 'r', encoding='utf-8') as f:
#     # with open('yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
#     line = f.readline()
#     count = 0
#     while line:
#         # print(line)
#         count += 1
#         if count % 1000000 == 0:
#             print(count)
#             # break
#
#         review = json.loads(line)
#
#         if review['user_id'] not in user_idx_str:
#             line = f.readline()
#             continue
#
#         review_content, score = review['text'], review['stars']
#
#         user_id, item_id = review['user_id'], review['business_id']
#         # # 将时间字符串转换为 datetime 对象
#         # datetime_obj = datetime.strptime(review['date'], "%Y-%m-%d %H:%M:%S")
#         # # 获取时间戳（秒级）
#         # timestamp = datetime.timestamp(datetime_obj)
#         r = {
#             'review': review_content,
#             'stars': score,
#             'business_id': review['business_id']
#         }
#         if user_id not in user_reviews:
#             user_reviews[user_id] = [r]
#         else:
#             user_reviews[user_id].append(r)
#
#         line = f.readline()
#
#         # if count % 100000 == 0:
#         #     print(count)
#         #     break
#
#     f.close()
# #     print(count)
# #     # print(Philadelphia_count)
# #
# #
# data_output = open('user_reviews_for_gra2.pkl', 'wb')
# pickle.dump({'user_reviews': user_reviews}, data_output)
# data_output.close()
# print('File: ', 'user_reviews_for_gra2.pkl', 'saved!')
#
#



with open('../user_reviews_for_gra2.pkl', 'rb') as f:
    user_reviews = pickle.load(f, encoding='latin-1')['user_reviews']
    f.close()

# print(user_reviews)

short_user_count = 0

for u in user_reviews:
    # if len(user_reviews[u])<30:
    short_user_count += len(user_reviews[u])

    if u == 'VOiJrfDciZzuOcK3cSPRWg':
        print(full_user_idx_ndcg[new_user_map[u]])
        print(user_reviews[u])
        print(len(user_reviews[u]))
        print(int(len(user_reviews[u]) * 0.8))

        print(user_reviews[u][-2: ])

    sentiments = user_reviews[u]
    test_len = int(len(sentiments) * 0.8)
    arr = []
    for i, s in enumerate(sentiments):
        if i == test_len:
            arr.append('||')
        review, score = s['review'], s['stars']
        if score > 3:
            arr.append('+')
        else:
            arr.append('-')

    print(u, ''.join(arr))

print('short_user_count:', short_user_count/len(user_reviews))