#!/usr/bin/env bash
set -euo pipefail

# Reproducible P0-3/P0-5 GPU pipeline.  Large source archives stay on the
# workstation/external disk; only the selected CityGML benchmark is required.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}/.."
export PYTHONPATH="${ROOT}"

MANIFEST="${ROOT}/data/plateau_multicity_manifest.json"
RESULTS="${ROOT}/results"
WORK_ROOT="${CIMFUSEMARK_WORK_ROOT:-/root/autodl-tmp/CIMFuseMark_p0/attack_work}"
DEVICE="${CIMFUSEMARK_DEVICE:-cuda}"
PYTHON="${CIMFUSEMARK_PYTHON:-python3}"
mkdir -p "${RESULTS}" "${WORK_ROOT}"

if [[ -n "${CIMFUSEMARK_WAIT_PID:-}" ]]; then
  while kill -0 "${CIMFUSEMARK_WAIT_PID}" 2>/dev/null; do
    sleep 10
  done
fi

if [[ ! -f "${RESULTS}/rgcn_multicity_separated.pt" ]]; then
  "${PYTHON}" "${ROOT}/train_robust_contrastive.py" \
    --manifest "${MANIFEST}" --device "${DEVICE}" \
    --output-prefix rgcn_multicity_separated
fi

if [[ ! -f "${RESULTS}/rgcn_multicity_base.pt" ]]; then
  "${PYTHON}" "${ROOT}/train_robust_contrastive.py" \
    --config "${ROOT}/configs/robust_contrastive_base.json" \
    --manifest "${MANIFEST}" --device "${DEVICE}" \
    --output-prefix rgcn_multicity_base
fi

if [[ ! -f "${RESULTS}/deepsets_multicity.pt" ]]; then
  "${PYTHON}" "${ROOT}/train_robust_contrastive.py" \
    --manifest "${MANIFEST}" --device "${DEVICE}" --relation-mode no_edges \
    --output-prefix deepsets_multicity
fi

if [[ ! -f "${RESULTS}/rgcn_multicity_personalized.pt" ]]; then
  "${PYTHON}" "${ROOT}/personalize_hash.py" \
    --checkpoint "${RESULTS}/rgcn_multicity_separated.pt" \
    --manifest "${MANIFEST}" --split test --background-split validation \
    --device "${DEVICE}" --output "${RESULTS}/rgcn_multicity_personalized.pt"
fi

for variant in separated base personalized; do
  checkpoint="${RESULTS}/rgcn_multicity_${variant}.pt"
  curves="${RESULTS}/multicity_${variant}_curves.json"
  open_set="${RESULTS}/multicity_${variant}_open_set.json"
  if [[ ! -f "${curves}" ]]; then
    "${PYTHON}" "${ROOT}/evaluate_robustness_curves.py" \
      --manifest "${MANIFEST}" --checkpoint "${checkpoint}" --split test \
      --device "${DEVICE}" --work-root "${WORK_ROOT}" --output "${curves}"
  fi
  "${PYTHON}" "${ROOT}/evaluate_open_set.py" \
    --manifest "${MANIFEST}" --checkpoint "${checkpoint}" \
    --registered-split test --calibration-split validation --curves "${curves}" \
    --device "${DEVICE}" --output "${open_set}"
done

if [[ ! -f "${RESULTS}/multicity_deepsets_curves.json" ]]; then
  "${PYTHON}" "${ROOT}/evaluate_robustness_curves.py" \
    --manifest "${MANIFEST}" --checkpoint "${RESULTS}/deepsets_multicity.pt" --split test \
    --device "${DEVICE}" --work-root "${WORK_ROOT}" \
    --output "${RESULTS}/multicity_deepsets_curves.json"
fi
"${PYTHON}" "${ROOT}/evaluate_open_set.py" \
  --manifest "${MANIFEST}" --checkpoint "${RESULTS}/deepsets_multicity.pt" \
  --registered-split test --calibration-split validation \
  --curves "${RESULTS}/multicity_deepsets_curves.json" --device "${DEVICE}" \
  --output "${RESULTS}/multicity_deepsets_open_set.json"

echo "P0 GPU pipeline complete: ${RESULTS}"
