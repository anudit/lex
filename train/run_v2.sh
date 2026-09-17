#!/usr/bin/env bash
# lex-lite (v2): train the full-precision teacher, distil the 28 KB student,
# then fine-tune it; the best checkpoint by real-bench score ends up in
# ./checkpoints_v2, which bundle_lex.py packages into lex/src.
#
#   teacher   full-precision model (not shipped)        -> checkpoints_v2_teacher
#   student   QAT student, live distillation, 40 epochs -> checkpoints_v2_student
#   finetune  two lower-LR rounds, keeps the best       -> checkpoints_v2
#
# Usage: ./run_v2.sh [teacher|student|finetune|all]   (default: all)
set -euo pipefail
cd "$(dirname "$0")"

DATASET=./corpus/dataset_v2
TEACHER_DIR=./checkpoints_v2_teacher
STUDENT_DIR=./checkpoints_v2_student
COMMON=(--dataset "$DATASET" --seed 20260916 --workers 4)
stage=${1:-all}

if [[ $stage == teacher || $stage == all ]]; then
  uv run --offline python -u train.py "${COMMON[@]}" \
    --full-precision --out-dir "$TEACHER_DIR" \
    --dim 160 --embed-dim 64 --head-hidden 192 --n-layers 4 \
    --film-rank 64 --erase-rank 16 --ctx-views full --scalar-bits 16 \
    --dropout 0.1 --epochs 12 --lr 2e-3 --calibration-fraction 0.2 \
    --mine-weak-languages 2>&1 | tee train_v2_teacher.log
fi

if [[ $stage == student || $stage == all ]]; then
  uv run --offline python -u train.py "${COMMON[@]}" \
    --out-dir "$STUDENT_DIR" --weight-budget 30000 \
    --epochs 40 --warmup-epochs 4 --lr 2e-3 --calibration-fraction 0.2 \
    --mine-weak-languages \
    --teacher-checkpoint "$TEACHER_DIR/best_model.pt" \
    --distill-weight 0.5 --distill-temperature 2.0 2>&1 | tee train_v2_student.log
fi

if [[ $stage == finetune || $stage == all ]]; then
  ./run_v2_finetune.sh "$STUDENT_DIR" ./checkpoints_v2
fi
