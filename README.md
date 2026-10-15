# Brain Age Prediction from Structural MRI

3D deep-learning models that predict a person's age from a T1-weighted
structural brain MRI scan ("brain age"), with a focus on comparing how a
transformer's token count affects both prediction accuracy and how well
that accuracy generalizes to an unseen clinical subgroup.

## Problem

Brain-age models are used as a health proxy: a predicted age far from a
person's actual age (a "brain age gap") has been associated with a range of
neurological and psychiatric conditions. For that proxy to be useful, a
model has to generalize beyond the population it was tuned on. This project
trains ten 3D architectures on the same data and evaluates each one on two
held-out groups — age- and sex-matched healthy controls, and participants
flagged for ADHD — to see whether prediction quality holds up consistently
across both, and whether a specific architectural choice (how many spatial
tokens a transformer attends over) affects that consistency.

**On terminology:** "sensitivity" in the title refers to how a model's
behavior differs on the ADHD subgroup relative to matched controls, not the
classification metric (true positive rate) that shares the name. This is a
regression task with a continuous age target, so all comparisons use
regression metrics (MAE, R², Pearson correlation) computed separately per
group.

## Dataset

T1-weighted structural MRI volumes from a longitudinal pediatric/adolescent
neuroimaging cohort, with age, sex, and ADHD-diagnosis status recorded at
each visit. Participants can have multiple visits and are grouped into
families with more than one enrolled member.

The dataset is access-restricted and **is not included in this repository**.
`src/data.py` expects:
- a `.mat` file (HDF5 / MATLAB v7.3 format) with per-row `iid` (participant
  id), `eid` (visit id), `fid` (family id), and a `7 x N` array `X` holding
  `[age, site_id, sex, ADHD1, ADHD2, ADHD3, ADHD4]`,
- a `.npy` file with one preprocessed MRI volume per row.

## Data splitting: avoiding leakage across visits and families

A naive split risks two kinds of leakage: the same participant's other
visits ending up in both train and test, and relatives (who share genetics
and environment) ending up on both sides of a split. `src/data.py` avoids
both by excluding at the **participant** and **family** level:

1. **ADHD test set** — every visit of every participant flagged ADHD at any
   timepoint.
2. Every family containing an ADHD participant is excluded from every other
   split.
3. **Healthy test set** — age- and sex-matched controls drawn only from the
   remaining ADHD-free pool, at a configurable ratio to the ADHD group
   (default 3:1).
4. **Train / validation** — the remaining participants, split 80/20 by
   family.

A leakage report is printed on every run, checking that no participant id or
family id appears in more than one split. An earlier version of this
pipeline matched controls at a single visit and removed only that one row
from the training pool, which let a control's *other* visits stay in
training — the same brain appearing on both sides of the split. This version
fixes that by excluding whole participants and families, not rows.

## Model architectures

Ten configurations, all predicting a single scalar (age):

| Model | Family | Conv blocks | Tokens | Parameters |
|---|---|---|---|---|
| A | Hybrid CNN → transformer | 5 (5 pooled) | 2 | 3.86M |
| B | Hybrid CNN → transformer | 5 (4 pooled) | 36 | 3.86M |
| C | Hybrid CNN → transformer | 5 (3 pooled) | 392 | 3.91M |
| D | Hybrid CNN → transformer | 4 (4 pooled) | 36 | 1.19M |
| E | Hybrid CNN → transformer | 3 (3 pooled) | 392 | 0.56M |
| F | Plain 3D CNN (wide head) | 5 | – | 1.46M |
| G | Plain 3D CNN | 6 | – | 2.99M |
| H | Plain 3D CNN | 4 | – | 0.32M |
| I | Plain 3D CNN | 6 | – | 4.79M |
| J | Plain 3D CNN (SFCN-style) | 6 | – | 2.97M |

The hybrid models (A–E) share the same transformer (2 layers, 4 heads,
embedding dim 128) and regression head. **A → B → C** is the cleanest
token-count comparison: same conv depth and almost the same parameter
count, with only the number of pooling stages changing. D and E reach the
same token counts as B and C by dropping conv blocks instead, so comparing
B vs. D and C vs. E separates token count from CNN depth and model size.
The plain CNNs (F–J) are a no-attention baseline.

Both families are built from small configurable blocks (`src/models.py`), so
an architecture is defined as a list of per-layer config dicts rather than
hardcoded layers — see `src/model_configs.py` for the exact configurations
compared.

## Training

Each of the 10 architectures is trained from 3 random seeds (150 epochs,
early stopping on validation loss):

- Loss: MSE · Optimizer: AdamW (lr 5e-6, weight decay 1e-3)
- Early stopping patience: 20 epochs · Gradient clipping: max norm 1.0
- Mixed-precision training on GPU

Training is implemented once in `src/engine.py::train` and reused across all
30 runs (10 models × 3 seeds). Each run's weights and loss history are saved
to a checkpoint (`save_model_with_metrics`) that can be reloaded for
evaluation without retraining.

## Evaluation

Every trained checkpoint is evaluated on the healthy test set and the ADHD
test set — both held out from training and model selection — using MAE, R²,
and Pearson correlation (`src/engine.py::predict`). Comparing these metrics
between the two groups, per architecture, is the core result: it shows
whether an architecture's accuracy is consistent across groups or degrades
on the clinical subgroup, and whether that pattern tracks the transformer's
token count.

## Tests

The dataset is restricted, so `tests/` checks the pipeline on synthetic data
instead:

- **Split integrity** (`tests/test_data.py`): on a synthetic multi-visit,
  multi-member-family cohort, no participant or family appears in more than
  one split; the ADHD test set holds every visit of every ADHD participant;
  ADHD families are kept out of every other split; and the split is
  reproducible for a fixed seed. One test also runs the full
  `.mat` + `.npy` loading path end to end.
- **Models and training** (`tests/test_models.py`): all ten architectures
  return one age per scan for a 3D input, the hybrid models produce
  the token counts listed above, the metric functions are correct on a
  known case, and a two-epoch CPU training run completes.

```bash
pytest tests
```

## Results

This repository ships the code, not pre-computed results or trained weights
— the source dataset is access-restricted, so results depend on running the
pipeline against your own copy of it. Running `notebooks/model_training_and_evaluation.ipynb`
end to end reproduces the full comparison (training curves, per-group
metrics table, and predicted-vs-actual plots) and writes it to `results/`.

## Repository structure

```
Hybrid-3D-brain-age/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── model_training_and_evaluation.ipynb   # full pipeline: data -> train -> evaluate -> analyze
└── src/
    ├── data.py            # leak-free participant/family-level train/val/test split
    ├── models.py           # ModularCNN3D and ModularHybridCNNTransformer architectures
    ├── model_configs.py    # the 10 architecture configurations (A-J) compared in this project
    └── engine.py           # train / predict / checkpoint / plotting utilities
└── tests/                  # synthetic-data tests for the split, models, and training loop
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for training (30 runs of up to
150 epochs each on 3D volumes); the notebook also supports a fast
`NUM_EPOCHS`-based smoke test on CPU to check the pipeline runs correctly
before committing to a full run.

## Running

1. Place your `.mat` and `.npy` files under `data/` (or set
   `BRAIN_AGE_DATA_DIR` to their location).
2. Open `notebooks/model_training_and_evaluation.ipynb` and run all cells.
3. For a quick correctness check before a full run, set `NUM_EPOCHS = 2` in
   the training-configuration cell.

## Limitations

- Three seeds per architecture is enough to see a trend, not enough for a
  significance test on the run-to-run variance.
- The healthy test set is matched to the ADHD group on age and sex only; it
  does not control for other variables (e.g. imaging site) that could also
  differ between groups.
- Parameter counts are matched only within A–C. D and E are much smaller,
  and the plain-CNN family (F–J) uses one convolution per block where the
  hybrid uses two, so compare across families with the parameter column in
  mind.

## License

Released under the [MIT License](LICENSE). The license covers the code
only. Use of the MRI data is governed by the original dataset's data use
agreement.
