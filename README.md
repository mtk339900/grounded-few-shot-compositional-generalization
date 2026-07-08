# Grounded Few-Shot English Learner

A synthetic-world testbed for studying **compositional generalization** in
sequence generation: a model must describe scenes ("a small red circle is
above a large blue star") built from `(object1, relation, object2)` triples,
and generalize to attribute combinations and object combinations it never
saw during training.

## Idea

Instead of asking "can a model memorize sentences," the benchmark asks
"can a model recombine parts it learned separately." Training data
deliberately withholds specific `(relation, shape)` and `(shape, shape)`
combinations; success is measured by exact-match accuracy on scenes built
from those withheld combinations.

## Architecture

The core model (`Stage3Model` / `Stage5Model` / `ScaledStage3Model`) has
three components:

1. **Factored slot encoder** — the scene vector is split into independent
   `shape` / `color` / `size` / `relation` fields *before* any mixing, each
   projected to its own embedding.
2. **Slot identity embeddings** — a learned positional tag added to each
   of the 7 slots, giving the decoder a way to bind "this shape belongs to
   object 1" vs. "object 2" (solves the attribute-binding problem).
3. **Attention decoder with supervised alignment** — an LSTM decoder with
   Bahdanau-style attention over the 7 slots; during training, attention is
   additionally supervised to point at the correct slot for each output
   word, then generates the sentence autoregressively at inference.

An auxiliary disentanglement loss (predict shape/color/size/relation from
each slot independently) is used as a regularizer.

## Experiment stages

| Stage | Question |
|---|---|
| 1 | Baseline: direct classification into fixed categories |
| 2 | Learned word embeddings instead of one-hot categories |
| 3 | Full sentence generation, no explicit concept labels |
| 4 | Paraphrase understanding (contrastive text-scene alignment) |
| 5 | Two-fact paragraphs |
| 6 | Single held-out (relation, shape) compositional split |
| 7 | GRU baseline (no attention, no factored encoding) |
| 8 | Multi-split compositional benchmark (8 independent splits) |
| 9 | Ablation study: factored encoder / attention supervision / slot identity |
| 10 | Multi-seed evaluation harness (mean ± std across seeds) |
| 11 | Transformer baselines (standard field-embedding vs. factored encoder) |
| 13 | World scaling: 4×4×2×4 → 25×25×20×25 vocabulary, plus a severe out-of-distribution split (held-out shapes never co-occur in training) |

## Results (reference run, seed=42)

- Stage 1 (direct classification): held-out exact-match accuracy = 1.000
- Stage 3 → Stage 8: the full factored+attention model reaches ~1.000
  compositional accuracy on held-out `(relation, shape)` combinations,
  averaged across 8 independent splits, vs. a GRU baseline around 0.3–0.4.
- Stage 9 ablations isolate which component drives that gap (see
  `run_ablation_study()` output for per-component contribution).

Exact numbers depend on the run; use `run_multi_seed_benchmark()` for
statistically robust mean ± std figures rather than a single seed.

## Repository layout

```
config.py          # all hyperparameters, world vocab, seeding
model.py            # the full script (Stages 1-13): models, training, evaluation
requirements.txt
README.md
```

## Running

```bash
pip install -r requirements.txt
python model.py
```

Running the whole file end-to-end (all 13 stages) is expensive; `__main__`
is currently set to run only Stage 13 (world scaling), since Stages 1-12
are already validated in prior runs. To re-run any earlier stage, call its
function directly, e.g.:

```python
from model import train_stage3, run_ablation_study, run_multi_seed_benchmark

model, train_data, held_data = train_stage3()
run_ablation_study(epochs=400)
run_multi_seed_benchmark(seeds=(0, 1, 2))
```

## Reproducibility note

`config.set_all_seeds()` seeds Python's `random`, `numpy`, and `torch`
(CPU and CUDA). Full determinism additionally requires
`torch.backends.cudnn.deterministic = True`, which trades off some GPU
throughput — set `torch.backends.cudnn.benchmark = True` instead if you
need speed and don't need bit-exact reproducibility across runs.

## Citation

If you use this benchmark, please cite:

```
@misc{grounded-fewshot-learner,
  title  = {Grounded Few-Shot English Learner: A Compositional Generalization Benchmark},
  year   = {2026},
  note   = {Synthetic scene-description benchmark for testing attribute binding
            and compositional generalization in sequence generation models.}
}
```
