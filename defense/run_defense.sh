#!/bin/bash
#
# run_defense.sh is a script which executes the defense
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_defense.sh INPUT_DIR OUTPUT_FILE
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_FILE - file to store classification labels
#
# Checkpoints are available at https://github.com/tensorflow/models/tree/master/slim
# and https://github.com/tensorflow/models/tree/master/adv_imagenet_models

INPUT_DIR=$1
OUTPUT_FILE=$2

#####
python defense_v4.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="/tmp/pred_6.csv" \
  --checkpoint_path=inception_v4.ckpt

python defense_pure_resnet_152.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="/tmp/pred_7.csv" \
  --checkpoint_path=resnet_v2_152.ckpt

#####
python defense_ens_adv.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="/tmp/pred_1.csv" \
  --checkpoint_path=ens_adv_inception_resnet_v2.ckpt


python defense_adv.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="/tmp/pred_2.csv" \
  --checkpoint_path=ens4_adv_inception_v3.ckpt



python defense_adv.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="/tmp/pred_3.csv" \
  --checkpoint_path=ens3_adv_inception_v3.ckpt


python defense_adv.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="/tmp/pred_4.csv" \
  --checkpoint_path=adv_inception_v3.ckpt


python defense_ens_adv.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="/tmp/pred_5.csv" \
  --checkpoint_path=inception_resnet_v2_2016_08_30.ckpt



python fool.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="/tmp/pred_fool.csv" \
  --checkpoint_path=inception_v3.ckpt


python defense_merge.py \
  --output_file="${OUTPUT_FILE}"


