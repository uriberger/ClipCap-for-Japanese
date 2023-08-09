#!/bin/sh

venv/bin/python eval_multiple.py reformulation_experiment/data/infer/base_infer_on_test_@0@.token.json \
                reformulation_experiment/data/infer/gt_infer_on_test_@0@_1_epoch.token.json \
                reformulation_experiment/data/infer/gt_infer_on_test_@0@_5_epoch.token.json
