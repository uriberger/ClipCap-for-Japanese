#!/bin/sh
set -e

MSG_PREFIX=[LOG_MSG]
BASE_DIR=reformulation_experiment
EXP_IND=2
BASE_SAMPLE_NUM=50000
ADD_SAMPLE_NUM=100000

echo "Experiment ${EXP_IND} part 3, base training sample num ${BASE_SAMPLE_NUM}, additional training sample num ${ADD_SAMPLE_NUM}"

# Reformulations based training
echo "$MSG_PREFIX Reformulations training"
venv/bin/python train.py --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_reformulated --epochs 5 --json_file ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_reformulated.json --image_ids_file ${BASE_DIR}/data/image_ids/additional_train_image_ids_${EXP_IND}.json --save_every 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --bs 10
echo "$MSG_PREFIX Reformulations inference 1 epoch"
venv/bin/python train.py --eval --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_reformulated/coco_prefix-000.pt --out_file ${BASE_DIR}/data/infer/reformulations_infer_on_test_${EXP_IND}_1_epoch
echo "$MSG_PREFIX Reformulations inference 5 epochs"
venv/bin/python train.py --eval --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_reformulated/coco_prefix-004.pt --out_file ${BASE_DIR}/data/infer/reformulations_infer_on_test_${EXP_IND}_5_epoch

echo "$MSG_PREFIX Finished"
