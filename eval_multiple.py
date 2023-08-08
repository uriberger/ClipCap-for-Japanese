import json
import os
import sys
from tqdm import tqdm
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


@dataclass
class Metrics:
    bleu: float
    rouge: float
    meteor: float
    cider: float
    spice: float


def compute_metrics(references, candidates, is_ja):
    ###BLEU#####
    print("Compute BLEU ... ")
    pycoco_bleu = Bleu()
    bleu, _ = pycoco_bleu.compute_score(references, candidates)

    ####METEOR###
    print("Compute METEOR ... ")
    pycoco_meteor = Meteor()
    meteor, _ = pycoco_meteor.compute_score(references, candidates)
    del pycoco_meteor
    # meteor = 0 # METEORはたまにバグるので

    ####ROUGE###
    print("Compute ROUGE ... ")
    pycoco_rouge = Rouge()
    rouge, _ = pycoco_rouge.compute_score(references, candidates)

    ####CIDER###
    print("Compute CIDER ... ")
    pycoco_cider = Cider()
    cider, _ = pycoco_cider.compute_score(references, candidates)

    ####SPICE####
    print("Compute SPICE ... ")
    if is_ja:
        spice = 0
    else:
        pycoco_spice = Spice()
        spice, _ = pycoco_spice.compute_score(references, candidates)

    metrics = Metrics(bleu, rouge, meteor, cider, spice)
    return metrics


def main():
    input_patterns = sys.argv[1:]
    no_tokenization = False
    if input_patterns[-1] == '--no_tokenization':
        no_tokenization = True
        input_patterns = input_patterns[:-1]
    data = {}
    for pattern in input_patterns:
        file_name = pattern.split('/')[-1]
        dir_path = '/'.join(pattern.split('/')[:-1])
        file_parts = file_name.split('@')
        assert len(file_parts) == 3
        options = file_parts[1].split(',')
        file_names = [file_parts[0] + x + file_parts[2] for x in options]
        file_paths = [os.path.join(dir_path, x) for x in file_names]
        
        data[pattern] = {}
        for file_path in file_paths:
            with open(file_path, 'r') as fp:
                data[pattern][file_path] = json.load(fp)

    with open(f"/cs/labs/oabend/uriber/datasets/STAIR-captions/stair_captions_v1.2_val_tokenized.json", "r") as f:
        gt = json.load(f)
    gt_annotations = gt["annotations"]

    with open('../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
        coco_data = json.load(fp)['images']
        image_ids = [x['cocoid'] for x in coco_data if x['split'] == 'test']
        image_ids_dict = {x: True for x in image_ids}
        gt_annotations = [x for x in gt_annotations if x['image_id'] in image_ids_dict]

    references = {}
    for anno in gt_annotations:
        img_id = anno["image_id"]
        references.setdefault(img_id, [])
        if len(references[img_id]) < 5:
            if no_tokenization:
                cap = ''.join(anno["tokenized_caption"].split())
                cap = ' '.join(cap)
            else:
                cap = anno["tokenized_caption"]
            references[img_id].append(cap)
            
    for pattern_key, pattern_value in data.items():
        res = defaultdict(list)
        for results in pattern_value.values():
            candidates = {}
            for i, elem in tqdm(enumerate(results)):
                img_id, cap = elem["image_id"], elem["caption"]
                if img_id not in candidates:
                    if no_tokenization:
                        cap = ''.join(cap.split())
                        cap = ' '.join(cap)
                    candidates[img_id] = [cap]

            metrics = compute_metrics(references, candidates, is_ja=True)
            for i in range(4):
                res[f'bleu{i+1}'].append(metrics.bleu[i])
            res['rouge'].append(metrics.rouge)
            res['meteor'].append(metrics.meteor)
            res['cider'].append(metrics.cider)
            res['spice'].append(metrics.spice)
        print('>>>>>>>>>>')
        print(pattern_key)
        for metric in res:
            if len(res[metric]) > 1:
                print(f'\t{metric}: {statistics.mean(res[metric])} +- {statistics.stdev(res[metric])}')
            else:
                print(f'\t{metric}: {res[metric][0]}')
        print('<<<<<<<<<<')

if __name__ == "__main__":
    main()
