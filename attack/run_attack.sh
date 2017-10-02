#!/bin/bash
#
# run_attack.sh is a script which executes the attack
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_attack.sh INPUT_DIR OUTPUT_DIR MAX_EPSILON
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_DIR - directory where adversarial images should be written
#   MAX_EPSILON - maximum allowed L_{\infty} norm of adversarial perturbation
#

INPUT_DIR=$1
OUTPUT_DIR=$2
MAX_EPSILON=$3

# Attack ensemble
NUM_ITERATIONS=9
ITER_ALPHA=2

python iter_fgsm_5_models.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --num_iter="${NUM_ITERATIONS}" \
  --iter_alpha="${ITER_ALPHA}" \
  --checkpoint_path1=ens_adv_inception_resnet_v2.ckpt \
  --checkpoint_path2=inception_v3.ckpt \
  --checkpoint_path3=nips_adv_inception_v3.ckpt \
  --checkpoint_path4=nips04_ens4_adv_inception_v3.ckpt \
  --checkpoint_path5=nips_inception_resnet_v2_2016_08_30.ckpt


