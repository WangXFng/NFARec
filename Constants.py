PAD = 0
# Yelp2023  Beauty Food.com Amazon-book ml-1M
DATASET = "Yelp2023"
ENCODER = 'THP'  # NHP THP Transformer
ABLATIONs = {'w/oWeSch', 'w/oPopDe', 'w/oSA', 'w/oNorm', 'w/oUSpec', 'w/oHgcn', 'w/oDisen'}
ABLATION = 'Full'


user_dict = {
    'ml-1M': 6041,
    'Beauty': 22363,
    'Yelp2023': 48993,
    'Food.com': 7452,
    'Amazon-book': 19804,
}


item_dict = {
    'ml-1M': 3955,
    'Beauty': 12101,
    'Yelp2023': 34298,
    'Food.com': 12911,
    'Amazon-book': 22086,
}

ITEM_NUMBER = item_dict.get(DATASET)
USER_NUMBER = user_dict.get(DATASET)

print('Dataset:', DATASET, '#User:', USER_NUMBER, '#POI', ITEM_NUMBER)
print('Encoder: ', ENCODER)
print('ABLATION: ', ABLATION)


DICT = {
    'ml-1M': 2,
    'Beauty': USER_NUMBER,
    'Yelp2023': USER_NUMBER,
    # 'Food.com': USER_NUMBER,
    # 'Amazon-book': USER_NUMBER,
}


NUM_LAYERS_DICT = {
    'ml-1M': 2,
    'Beauty': 2,
    'Yelp2023': 2,
    'Food.com': 1,
    'Amazon-book': 1,
}

NUM_LAYERS = NUM_LAYERS_DICT[DATASET]



