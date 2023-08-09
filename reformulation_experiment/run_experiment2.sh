#!/bin/sh
set -e

MSG_PREFIX=[LOG_MSG]
BASE_DIR=reformulation_experiment
EXP_IND=2
BASE_SAMPLE_NUM=50000
ADD_SAMPLE_NUM=100000

echo "Experiment ${EXP_IND} part 2, base training sample num ${BASE_SAMPLE_NUM}, additional training sample num ${ADD_SAMPLE_NUM}"

# Own captions based training
echo "$MSG_PREFIX Own captions training"
venv/bin/python train.py --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_own --epochs 5 --json_file ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}.token.json --image_ids_file ${BASE_DIR}/data/image_ids/additional_train_image_ids_${EXP_IND}.json --save_every 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --bs 10
echo "$MSG_PREFIX Own captions inference 1 epoch"
venv/bin/python train.py --eval --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_own/coco_prefix-000.pt --out_file ${BASE_DIR}/data/infer/own_infer_on_test_${EXP_IND}_1_epoch
echo "$MSG_PREFIX Own captions inference 5 epochs"
venv/bin/python train.py --eval --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_own/coco_prefix-004.pt --out_file ${BASE_DIR}/data/infer/own_infer_on_test_${EXP_IND}_5_epoch

# Reformulations based training
echo "$MSG_PREFIX ja->en"
venv/bin/python translate.py --input_file ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}.json --output_file ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_en_translated --source_language ja --target_language en --output_format caption
echo "$MSG_PREFIX Reformulation"
cd ../AliceMind/mPLUG
rm -f ../../ClipCap-for-Japanese/${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_en_reformulated.json
venv/bin/python reformulate.py --model_path output/vqa_mplug_base/checkpoint_07.pth --input_file ../../ClipCap-for-Japanese/${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_en_translated.json --output_format caption --output_file ../../ClipCap-for-Japanese/${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_en_reformulated --dataset COCO
cd ../../ClipCap-for-Japanese
echo "$MSG_PREFIX en->ja"
venv/bin/python translate.py --input_file ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_en_reformulated.json --output_file ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_reformulated --source_language en --target_language jap --output_format caption

echo "$MSG_PREFIX Finished"
