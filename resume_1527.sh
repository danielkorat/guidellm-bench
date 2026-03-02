#!/bin/bash
# Temporary script: completes the 20260302_1527 experiment.
#
# Skips the 4 already-successful configs (checkpoint files present):
#   ✓  openai_gpt-oss-20b_tp2_quant-none       (bad data / OOM, but JSON exists — skipped)
#   ✓  Qwen_Qwen3-30B-A3B_tp4_quant-fp8
#   ✓  Qwen_Qwen3-4B-Thinking-2507_tp2_quant-fp8
#   ✓  Qwen_Qwen3-4B-Thinking-2507_tp4_quant-fp8
#
# Will run only the 2 missing configs:
#   →  openai_gpt-oss-20b_tp4_quant-none       (failed: GPU corrupted by prior OOM)
#   →  Qwen_Qwen3-30B-A3B_tp2_quant-fp8        (failed: GPU corrupted by prior OOM)
#
# PREREQUISITE: host reboot to clear GPU state from gpt-oss-20b_tp2 OOM crash.
#   ssh root@10.75.137.163 && reboot
#
# max-seconds is now 900 (bumped from 600 — Qwen3-30B needs ~540s for 30 × 18s reqs).
#
# Run as:
#   bash resume_1527.sh
#   # then: tail -f guidellm_results/20260302_1527/bench.log

set -e
cd "$(dirname "$(readlink -f "$0")")"

echo "Resuming guidellm_results/20260302_1527 ..."
nohup ./bench.py --resume guidellm_results/20260302_1527 > /dev/null 2>&1 &
echo "PID $!  —  tail -f guidellm_results/20260302_1527/bench.log"
