import json
import argparse
from lib2to3.pgen2.tokenize import tokenize
import MeCab
from tqdm import tqdm
from dataclasses import dataclass
from functools import reduce
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--image_ids_file', type=str)
    parser.add_argument('--evaluate_on_val', action='store_true')
    args = parser.parse_args()

    results_dir = args.reuslts_dir

    results, gt = [], None
    with open(f"./{results_dir}/eval.json", "r") as f:
        results = json.load(f)

    with open(f"/cs/labs/oabend/uriber/datasets/STAIR-captions/stair_captions_v1.2_val_tokenized.json", "r") as f:
        gt = json.load(f)
    gt_annotations = gt["annotations"]

    image_ids = None

    if args.image_ids_file is not None:
        with open(args.image_ids_file, 'r') as fp:
            image_ids = json.load(args.image_ids_file)
    elif not args.evaluate_on_val:
        # If the user didn't explicitly asked us to evaluate to the validation set, we evaluate only on the test set,
        # according to karpathy's splits
        with open('../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
            coco_data = json.load(fp)['images']
            image_ids = [x['cocoid'] for x in coco_data if x['split'] == 'test']

    if image_ids is not None:
        image_ids_dict = {x: True for x in image_ids}
        gt_annotations = [x for x in gt_annotations if x['image_id'] in image_ids_dict]

    tagger = MeCab.Tagger("-Owakati")
    candidates = {}
    img_set = set()
    for i, elem in tqdm(enumerate(results)):
        img_id, cap = elem["image_id"], elem["caption"]
        if img_id not in candidates:
            tokenized = ' '.join(tagger.parse(cap).split())
            candidates[img_id] = [tokenized]
            img_set.add(img_id)

    references = {}
    for anno in gt_annotations:
        img_id = anno["image_id"]
        if img_id not in img_set:
            continue
        references.setdefault(img_id, [])
        if len(references[img_id]) < 5:
            references[img_id].append(anno["tokenized_caption"])

    # print(candidates)
    metrics = compute_metrics(references, candidates, is_ja=True)
    print(metrics)


if __name__ == "__main__":
    main()
