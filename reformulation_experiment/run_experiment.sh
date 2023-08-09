#!/bin/sh
set -e

MSG_PREFIX=[LOG_MSG]
BASE_DIR=reformulation_experiment
EXP_IND=0
BASE_SAMPLE_NUM=30000
ADD_SAMPLE_NUM=100000

echo "Experiment ${EXP_IND} part 1, base training sample num ${BASE_SAMPLE_NUM}, additional training sample num ${ADD_SAMPLE_NUM}"

# Base training
echo "$MSG_PREFIX Prepare base training data"
venv/bin/python ${BASE_DIR}/prepare_base_training_data.py ${EXP_IND} ${BASE_SAMPLE_NUM}
echo "$MSG_PREFIX Base training"
venv/bin/python train.py --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_base --epochs 10 --image_ids_file ${BASE_DIR}/data/image_ids/base_train_image_ids_${EXP_IND}.json --save_every 10
echo "$MSG_PREFIX Base inference"
venv/bin/python train.py --eval --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --out_file ${BASE_DIR}/data/infer/base_infer_on_test_${EXP_IND}

# GT based training
echo "$MSG_PREFIX Prepare gt training data"
venv/bin/python ${BASE_DIR}/prepare_gt_training_data.py ${EXP_IND} ${ADD_SAMPLE_NUM}
echo "$MSG_PREFIX GT training"
venv/bin/python train.py --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_gt --epochs 5 --json_file ${BASE_DIR}/data/gt_train_data/train_data_${EXP_IND}.json --save_every 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt
echo "$MSG_PREFIX GT inference 1 epoch"
venv/bin/python train.py --eval --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_gt/coco_prefix-000.pt --out_file ${BASE_DIR}/data/infer/gt_infer_on_test_${EXP_IND}_1_epoch
echo "$MSG_PREFIX GT inference 5 epochs"
venv/bin/python train.py --eval --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_gt/coco_prefix-004.pt --out_file ${BASE_DIR}/data/infer/gt_infer_on_test_${EXP_IND}_5_epoch

# Translation based training
echo "$MSG_PREFIX Translation training"
venv/bin/python train.py --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_translated --epochs 5 --json_file ${BASE_DIR}/data/translated_data/coco_ja_translated_helsinki.json --image_ids_file ${BASE_DIR}/data/image_ids/additional_train_image_ids_${EXP_IND}.json --save_every 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --bs 16
echo "$MSG_PREFIX Translation inference 1 epoch"
venv/bin/python train.py --eval --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_translated/coco_prefix-000.pt --out_file ${BASE_DIR}/data/infer/translated_infer_on_test_${EXP_IND}_1_epoch
echo "$MSG_PREFIX Translation inference 5 epochs"
venv/bin/python train.py --eval --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_translated/coco_prefix-004.pt --out_file ${BASE_DIR}/data/infer/translated_infer_on_test_${EXP_IND}_5_epoch

# Base inference for own and reformulation based training
echo "$MSG_PREFIX Base inference on val"
venv/bin/python train.py --eval --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --json_file /cs/labs/oabend/uriber/datasets/STAIR-captions/stair_captions_v1.2_train_tokenized.json --image_ids_file ${BASE_DIR}/data/image_ids/additional_train_image_ids_${EXP_IND}.json --out_file ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}

echo "$MSG_PREFIX Finished"
