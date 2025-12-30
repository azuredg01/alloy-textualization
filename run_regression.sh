#!/usr/bin/env bash
set -euo pipefail

# 共用參數
GPU_CFG="gpu_yaml/default_config.yaml"
SCRIPT="main_DDP_v3.py"
ACCEL="$(which accelerate)"

# 要跑的資料集清單
training_config=(
  config_regression.yaml
  config_calcDensity_A-D+HV.yaml
  config_calcDensity_A-D+G.yaml
  config_calcDensity_A-D+G+HV.yaml
  config_ExpDensity_A-D.yaml
  config_ExpDensity_A-D+HV.yaml
  config_ExpDensity_A-D+G.yaml
  config_ExpDensity_A-D+G+HV.yaml
)

# ---- 讀一次 sudo 密碼，並包成 sdo() ----
read -rsp "sudo password: " SUDOPASS; echo
sdo() { printf '%s\n' "$SUDOPASS" | sudo -S "$@"; }
# 可選：先驗證並建立 sudo 快取，避免第一次在指令中才驗證
printf '%s\n' "$SUDOPASS" | sudo -Sv >/dev/null 2>&1 || true
trap 'unset -v SUDOPASS' EXIT

# 逐一跑
for tcf in "${training_config[@]}"; do
  sdo "$ACCEL" launch --config_file "$GPU_CFG" "$SCRIPT" \
    -tcf "config/$tcf"
done

