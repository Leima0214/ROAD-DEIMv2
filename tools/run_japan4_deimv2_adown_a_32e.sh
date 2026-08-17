#!/usr/bin/env bash
set -uo pipefail

cd /root/ROAD-DEIMv2 || exit 90
ulimit -n 65535
mkdir -p logs reports

run_name="deimv2_n_japan4_adown_a_32e_seed42_20260817"
output_dir="outputs/${run_name}"
log_path="logs/${run_name}.log"
exit_path="logs/${run_name}.exitcode"
preflight_report="reports/${run_name}_preflight.json"
preflight_log="logs/${run_name}_preflight.log"
checkpoint="weights/deimv2_hgnetv2_n_coco_hf.pth"
base_config="configs/deimv2/deimv2_hgnetv2_n_japan4.yml"
adown_config="configs/deimv2/deimv2_hgnetv2_n_japan4_adown_a_32e.yml"

if [[ -e "${output_dir}" || -e "${log_path}" || -e "${exit_path}" \
      || -e "${preflight_report}" || -e "${preflight_log}" ]]; then
  echo "ADOWN_A_LAUNCH_REFUSED: target artifacts already exist" >&2
  exit 91
fi
for required in "${checkpoint}" "${base_config}" "${adown_config}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ADOWN_A_LAUNCH_REFUSED: missing ${required}" >&2
    exit 92
  fi
done

CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python -u \
  tools/preflight_japan4_deimv2_adown_a.py \
  --base-config "${base_config}" \
  --adown-config "${adown_config}" \
  --checkpoint "${checkpoint}" \
  --report "${preflight_report}" \
  --device cuda:0 --batch-size 2 2>&1 | tee "${preflight_log}"
preflight_status=${PIPESTATUS[0]}
if [[ "${preflight_status}" -ne 0 ]]; then
  printf '%s\n' "${preflight_status}" > "${exit_path}"
  echo "ADOWN_A_LAUNCH_REFUSED: dynamic preflight failed" >&2
  exit "${preflight_status}"
fi

CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python -u train.py \
  -c "${adown_config}" \
  -t "${checkpoint}" \
  --use-amp --seed 42 -d cuda:0 \
  --output-dir "${output_dir}" 2>&1 | tee "${log_path}"
status=${PIPESTATUS[0]}
printf '%s\n' "${status}" > "${exit_path}"
exit "${status}"
