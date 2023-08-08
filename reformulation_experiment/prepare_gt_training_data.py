import json
import sys
import random
from collections import defaultdict

assert len(sys.argv) == 2 or len(sys.argv) == 3
exp_ind = int(sys.argv[1])

with open(f'reformulation_experiment/data/image_ids/additional_train_image_ids_{exp_ind}.json', 'r') as fp:
    image_ids = json.load(fp)

if len(sys.argv) == 3:
    add_train_sample_num = int(sys.argv[2])
    image_ids = random.sample(image_ids, add_train_sample_num)
image_ids_dict = {x: True for x in image_ids}

with open('/cs/labs/oabend/uriber/datasets/STAIR-captions/stair_captions_v1.2_train.json', 'r') as fp:
    train_data = json.load(fp)['annotations']

image_id_to_captions = defaultdict(list)
for x in train_data:
    if x['image_id'] in image_ids_dict:
        image_id_to_captions[x['image_id']].append(x['tokenized_caption'])
        
res = []
for image_id, captions in image_id_to_captions.items():
    res.append({'image_id': image_id, 'tokenized_caption': random.choice(captions)})

train_data = {'annotations': res}
    
with open(f'reformulation_experiment/data/gt_train_data/train_data_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(train_data))
