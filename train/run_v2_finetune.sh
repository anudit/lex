#!/usr/bin/env bash
# Fine-tune a lex-lite (v2) student at decreasing learning rates, still
# distilling from the v2 teacher, and copy the best checkpoint -- scored exactly
# as it ships -- to <dest>. Each round resumes from the best so far and only
# saves when it beats its own starting score.
#
# Usage: ./run_v2_finetune.sh <source-dir> <dest-dir>
set -euo pipefail
cd "$(dirname "$0")"

SRC=${1:?source checkpoint dir}
DEST=${2:?destination dir}
LRS=(${LRS:-5e-4 2.5e-4})
EPOCHS=${EPOCHS:-16}

score() {
  uv run --offline python eval_real_bench.py --checkpoint "$1/best_model.pt" 2>/dev/null \
    | sed -n 's/^real-bench weighted accuracy: *\([0-9.]*\)%.*/\1/p'
}

best_dir=$SRC
best=$(score "$best_dir")
echo "FINETUNE: $best_dir scores $best%"
round=0
for lr in "${LRS[@]}"; do
  round=$((round + 1))
  out=./checkpoints_v2_ft$round
  echo "FINETUNE: round $round resume $best_dir lr $lr -> $out"
  uv run --offline python -u train.py --dataset ./corpus/dataset_v2 \
    --seed $((20260916 + round)) --workers 4 --out-dir "$out" \
    --resume "$best_dir/best_model.pt" --epochs "$EPOCHS" --lr "$lr" \
    --calibration-fraction 0.3 --mine-weak-languages --weight-budget 30000 \
    --teacher-checkpoint ./checkpoints_v2_teacher/best_model.pt \
    --distill-weight 0.5 --distill-temperature 2.0 2>&1 | tee "train_v2_ft$round.log"
  if [[ -f "$out/best_model.pt" ]]; then
    s=$(score "$out")
    echo "FINETUNE: $out scores $s%"
    if uv run --offline python -c "import sys; sys.exit(0 if $s > $best else 1)"; then
      best_dir=$out; best=$s
    fi
  fi
done

if [[ "$best_dir" != "$DEST" ]]; then
  rm -rf "$DEST" && cp -R "$best_dir" "$DEST"
fi
echo "FINETUNE DONE: $DEST (from $best_dir) real-bench $best%"
