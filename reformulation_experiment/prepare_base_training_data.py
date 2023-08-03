import json
import sys
import random
from collections import defaultdict

assert len(sys.argv) == 3
exp_ind = int(sys.argv[1])
train_sample_num = int(sys.argv[2])

with open('/cs/labs/oabend/uriber/datasets/STAIR-captions/stair_captions_v1.2_train.json', 'r') as fp:
    train_data = json.load(fp)['annotations']
train_image_ids = [x['image_id'] for x in train_data]
chosen_image_ids = random.sample(train_image_ids, train_sample_num)

with open(f'reformulation_experiment/data/image_ids/train_image_ids_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(chosen_image_ids))
