#!/bin/bash
# Orchestrates the remaining screen stages (resume-friendly).
# 1) abliterated 0.6B row (build + screen + metrics)
# 2) FULL stage: resumes 0.6B rows from checkpoints, computes 1.7B/4B families
#    (download -> screen -> self-abliterate -> screen), then the selection rule
#    and method_out.json.
set -u
cd "$(dirname "$0")"
export HF_HOME="$(pwd)/cache/hf_home"
LOG=logs_stage_full.log
echo "[$(date +%H:%M:%S)] full stage start" >> $LOG
nohup timeout 8700 .venv/bin/python method.py --stage full --splice-layers 5 > $LOG 2>&1 &
echo "full PID=$!"