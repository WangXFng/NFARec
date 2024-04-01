
data_names = ['Amazon-book', 'Beauty', 'Food.com', 'ml-1M', 'Yelp2023']
file_names = ['{}/{}_train.txt', '{}/{}_test.txt', '{}/{}_tune.txt']

negative_count, positive_count = 0, 0
for data_name in data_names:
    for file_name in file_names:
        with open(file_name.format(data_name, data_name)) as f:
            for line in f.readlines():
                user_id, item_id, score, time = line.replace('\r', '').split('\t')
                if int(score) < 4:
                    negative_count += 1
                else:
                    positive_count += 1

    print(data_name, ':', 'pos', positive_count/(negative_count+positive_count), "neg", negative_count/(negative_count+positive_count), )
