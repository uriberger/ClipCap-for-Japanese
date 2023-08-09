import sys
import os
import json
import MeCab
from tqdm import tqdm

def clean_sentence(sentence, fp):
    res = ''
    for x in sentence:
        try:
            fp.write(x)
            res += x
        except UnicodeEncodeError:
            continue
    return res

assert len(sys.argv) == 3
input_file = sys.argv[1]
split = sys.argv[2]

with open(input_file, 'r') as fp:
    input_data = json.load(fp)

tagger = MeCab.Tagger("-Owakati")
res = []
dummy_file_name = 'dummy.txt'
with open(dummy_file_name, 'w') as fp:
    for sample in tqdm(input_data):
        tokenized_caption = ' '.join(tagger.parse(sample['caption']).split())
        tokenized_caption = clean_sentence(tokenized_caption, fp)
        res.append({'image_id': sample['image_id'], 'tokenized_caption': tokenized_caption})
if os.path.isfile(dummy_file_name):
    os.remove(dummy_file_name)

output_file_name = input_file.split('.json')[0] + '.token.json'
if split == 'train':
    res = {'annotations': res}
with open(output_file_name, 'w') as fp:
    fp.write(json.dumps(res))
