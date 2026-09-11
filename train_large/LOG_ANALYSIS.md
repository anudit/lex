# Training log analysis

## What the completed runs establish

| run | packed | best real benchmark | best epoch | ending train accuracy |
| --- | ---: | ---: | ---: | ---: |
| hero5 | 35.74 KiB | 83.02% | 37 | 87.9% |
| hero6 | 36.40 KiB | 81.40% | 16 | 87.8% |
| hero7 | 35.74 KiB | 82.42% | 26 | 87.9% |
| selective52 | 36.15 KiB | **86.04%** | 34 | 92.4% |

The selective run is the only architectural change in this group that improved
both optimization and out-of-repository generalization. Its rank-8 erase path,
rank-32 FiLM projection, boundary loss, tempered sampling, and removal of five
low-value training languages improved the best real benchmark by 3.02 points
over hero5. Hero6 and hero7 are warnings against selecting architecture from the
internal validation curve alone: both continued to learn that split while their
real benchmark remained below hero5.

The full-precision-to-QAT transition is still the dominant optimization shock.
On selective52, training accuracy fell from 90.9% at epoch 4 to 83.2% at epoch 5,
then recovered steadily. A larger model should therefore keep a real FP warmup,
distill from an FP teacher, and checkpoint only quantized epochs.

The curves are not exhausted at epoch 20, but they are close to capacity-limited
by epoch 30. Selective52 rose from 83.58% real at its first QAT epoch to 85.77% at
epoch 28. The final natural-popularity phase raised training accuracy abruptly
from 89.4% to 92.0%, while validation stayed near 89.5%; this is useful
calibration, not evidence that more identical epochs will close the gap.

Boundary quality tracks overall quality closely and reached 89.57% at the chosen
checkpoint. The remaining error is therefore not mostly delayed transitions.
For 193 languages the next budget should go to lexical collision reduction,
more shared width, stronger file conditioning, and enough per-language data,
while retaining selective erase rather than replacing the recurrence again.

## Consequences for the large model

- Preserve the proven quantization recipe and selective erase.
- Increase the primary/secondary word hashes from 512/128 to 1024/256.
- Use width 96 and four layers rather than width 128 and three layers: the fourth
  dilation scale gives better syntax-depth coverage across dissimilar grammars.
- Double FiLM and erase ranks to stop 193 grammars competing for one undirected
  transformation.
- Add deterministic structural fields for about 1 KiB instead of asking learned
  recurrence channels to reconstruct bracket, indentation, and quote state.
- Select on a metric with a uniform coverage component. The old real benchmark
  only contains the compact language set and cannot choose a 193-language model.
- Use a full-precision teacher and logit distillation to soften inconsistent
  labels from two grammar engines before 1-bit QAT.
