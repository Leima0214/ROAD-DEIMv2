#!/usr/bin/env bash
set -uo pipefail

cd /root/ROAD-DEIMv2-M2 || exit 90
ulimit -n 65535
mkdir -p logs

run_name="deimv2_n_japan4_m2_ease_l2_32e_seed42_20260814"
output_dir="outputs/${run_name}"
log_path="logs/${run_name}.log"
exit_path="logs/${run_name}.exitcode"

if [[ -e "${output_dir}" || -e "${log_path}" || -e "${exit_path}" ]]; then
  echo "M2_LAUNCH_REFUSED: target artifacts already exist" >&2
  exit 91
fi

CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python -u train.py \
  -c configs/deimv2/deimv2_hgnetv2_n_japan4_m2_ease_l2_32e.yml \
  -t weights/deimv2_hgnetv2_n_coco_hf.pth \
  --use-amp --seed 42 -d cuda:0 \
  --output-dir "${output_dir}" 2>&1 | tee "${log_path}"
status=${PIPESTATUS[0]}
printf '%s\n' "${status}" > "${exit_path}"
exit "${status}"
