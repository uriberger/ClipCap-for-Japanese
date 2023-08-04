import sys
import json
import MeCab
from tqdm import tqdm

assert len(sys.argv) == 2
input_file = sys.argv[1]

with open(input_file, 'r') as fp:
    input_data = json.load(fp)

tagger = MeCab.Tagger("-Owakati")
res = []
for sample in tqdm(input_data):
    res.append({'image_id': sample['image_id'], 'caption': ' '.join(tagger.parse(sample['caption']).split())})

output_file_name = input_file.split('.json')[0] + '.token.json'
with open(output_file_name, 'w') as fp:
    fp.write(json.dumps(res))
