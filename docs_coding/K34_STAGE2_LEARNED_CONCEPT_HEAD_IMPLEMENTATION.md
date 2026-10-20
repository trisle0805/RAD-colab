# K34 — Stage 2 Learned Concept Head Implementation Request

Repository: `trisle0805/RAD-colab`
Branch: `WS`

## Goal

Replace the current Stage-2 `uniform mean + gamma_d + b_d` diagnosis layer with **learned linear concept heads** that still enforce the hard scalar proposition bottleneck.

This implementation must preserve the central thesis constraint:

> Diagnosis is computed only from scalar proposition evidence scores.

Do NOT allow diagnosis to read:
- ResNet features,
- BioClinicalBERT embeddings,
- cross-attention hidden vectors,
- proposition embeddings,
- any latent representation other than scalar proposition evidence.

Stage 1 is considered frozen and complete.

The official Stage-2 inputs will be the Stage-1 text-support-retrained score files:

```text
scores_train_best.npz
scores_val_best.npz
scores_test_best.npz
```

Their paths are supplied by the notebook / CLI. Do not hard-code any Drive or Colab path in repository code.

---

# 1. Models to implement

Implement two new Stage-2 diagnosis heads:

## S2-L — Local guideline-constrained head

Each disease `d` may read ONLY propositions belonging to disease `d`.

Use polarity-adjusted evidence:

```python
if polarity_j == +1:
    s_tilde_j = s_j
elif polarity_j == -1:
    s_tilde_j = 1 - s_j
```

The disease logit is:

```text
logit_d = b_d + sum_j w_dj * s_tilde_j
```

where `j` is restricted to propositions of disease `d`.

### Required sign constraint

Weights MUST satisfy:

```text
w_dj >= 0
```

Suggested implementation:

```python
w_dj = softplus(theta_dj)
```

or another smooth nonnegative parameterization.

This is important scientifically:

> Every active S2-L contribution must mean “evidence supporting disease d in the guideline-consistent direction”.

S2-L must NOT use negative weights.

---

## S2-G — Global sparse concept head

Each disease `d` may read ALL scalar proposition scores.

Use the ORIGINAL scalar scores:

```text
s_j in [0,1]
```

Do NOT polarity-flip inputs for S2-G.

Disease logit:

```text
logit_d = b_d + sum_j w_dj * s_j
```

with unrestricted signed weights:

```text
w_dj in R
```

Interpretation:

```text
w_dj > 0  -> proposition supports disease d
w_dj < 0  -> proposition reduces evidence for disease d
```

S2-G must support:
- L2 regularization;
- L1 regularization.

Do not standardize proposition scores before the head. Keep them on their original `[0,1]` scale.

---

# 2. Routing modes A / C / B

Both S2-L and S2-G must support the already-defined Stage-2 routing modes.

Reuse existing Stage-2 routing code where possible.

## A — `image_morphology`

Use only morphology propositions with image evidence.

## C — `full_image_morphology_text_other`

- morphology propositions use image score;
- non-morphology propositions use text score when available.

## B — `full_fused`

Use the existing Stage-1 scalar fusion rule:

- proposition with both image and text -> fused score;
- image-only proposition -> image score;
- text-only proposition -> text score.

Do not invent a new fusion rule in this task.

---

# 3. Diagnosis task and loss

SkinCAP diagnosis is a 47-class single-label task.

Use:

```python
CrossEntropyLoss
```

on the 47 logits.

Do NOT use 47 independent sigmoid BCE heads for the new Stage-2 concept head.

The softmax probability is used for diagnosis metrics.

The raw logits and proposition-level contributions must also be exported.

---

# 4. Required CLI

Add a new CLI entry point:

```bash
python -m proposition.cli.stage2_concept
```

It must support at least:

```text
--train_scores
--val_scores
--test_scores

--active_csv

--train_csv
--val_csv
--test_csv

--output_dir
--device
--seed

--head_mode
    local
    global

--routing_mode
    image_morphology
    full_image_morphology_text_other
    full_fused

--regularization
    none
    l1
    l2

--lambda_reg

--epochs
--patience
--learning_rate
--batch_size

--nonzero_threshold
```

No hard-coded filesystem paths in repository code.

Suggested default:

```text
nonzero_threshold = 1e-4
```

This threshold is for AUDIT/reporting only.

---

# 5. G-B-L2 replication gate

This is a mandatory implementation verification.

Before L1 sparsity experiments are trusted, the new Stage-2 implementation must be able to reproduce the previous scalar linear diagnostic probe.

The reference probe was:

```text
feature set: B / full_fused
classifier: multinomial logistic regression
selected C: 0.1
test Top-1 approximately 56.8%
val Top-1 approximately 60.3%
```

Important:

`C=0.1` in sklearn must NOT be naively equated to an arbitrary PyTorch `weight_decay`.

Instead, implement a dedicated reference mode or verification script that uses the SAME sklearn `LogisticRegression` contract as the previous probe:

```python
Pipeline([
    ("scale", StandardScaler()),   # only for the replication gate
    ("clf", LogisticRegression(
        C=0.1,
        solver="lbfgs",
        max_iter=5000,
        random_state=seed,
    )),
])
```

This reference gate is allowed to standardize because it exists ONLY to reproduce the previous diagnostic probe.

The production S2-L / S2-G heads MUST use raw `[0,1]` proposition scores without standardization.

Add a CLI or helper such as:

```bash
python -m proposition.cli.stage2_concept_reference
```

or a clearly separated function that the notebook can call.

The gate should export:

```text
val_top1
val_top3
test_top1
test_top3
macro_f1
macro_auc
mrr
cross_entropy
selected_C
```

Acceptance rule:

> If G-B-L2/reference cannot reproduce the previous probe within a small tolerance, the learned Stage-2 implementation must NOT proceed to L1 sweeps.

Use a sensible tolerance, e.g.:

```text
absolute Top-1 difference <= 0.01
```

for both val and test.

Do not silently continue if the gate fails.

---

# 6. Validation-only model selection

Test must not be used to choose:
- head type;
- routing mode;
- regularization;
- lambda;
- epoch;
- checkpoint.

All choices use validation only.

Because validation has only ~297 cases, implement the following selection policy in notebook-facing utilities or summary code.

## Practical-tie rule

1. Compute best validation Top-1.
2. Any configuration within:

```text
0.02 absolute Top-1
```

of the best is considered practically tied.

3. Among practically tied configurations:
   - prefer S2-L over S2-G;
   - within the same architecture, prefer the more regularized / simpler model;
   - if still tied, choose lower validation cross-entropy.

This preference logic should be implemented as a deterministic helper, not manually applied later.

Example helper:

```python
select_stage2_configuration(...)
```

It should return:
- selected run;
- reason;
- all tied candidates;
- validation metric values.

Do NOT inspect test values inside this selector.

---

# 7. Regularization grid support

Repository code only needs to support arbitrary `lambda_reg`.

The notebook will use a deliberately small grid.

Expected initial L1 grid:

```text
1e-5
1e-4
1e-3
1e-2
```

Do not hard-code this grid into the model.

For S2-G:
- support `regularization=l1`;
- support `regularization=l2`.

For S2-L:
- support at least `none` and `l2`;
- L1 support is optional but acceptable.

If L1 is implemented using standard penalty:

```text
loss = cross_entropy + lambda * |W|_1
```

that is acceptable.

Proximal L1 / exact zeros are optional, not required.

---

# 8. Contribution semantics and exports

This is critical for XAI.

For every case `i`, disease `d`, proposition `j`, define contribution:

## S2-L

```text
contribution_i,d,j = w_dj * s_tilde_i,j
```

Only disease-local propositions are allowed.

All active contributions should be nonnegative.

## S2-G

```text
contribution_i,d,j = w_dj * s_i,j
```

Contribution may be positive or negative.

The exact disease logit reconstruction must satisfy:

```text
logit_i,d =
    bias_d
    + sum_j contribution_i,d,j
```

within numerical tolerance.

Add a unit test for exact reconstruction.

---

# 9. Top-1 vs Top-2 explanation

Viewer support must be prepared now, even though the HTML viewer itself is NOT part of this task.

For each case, export:

```text
top1_disease
top2_disease

top1_logit
top2_logit

margin = top1_logit - top2_logit
```

Also export proposition-level margin contribution:

```text
delta_contribution_j
=
contribution_top1_j
-
contribution_top2_j
```

It must satisfy:

```text
margin
=
(bias_top1 - bias_top2)
+
sum_j delta_contribution_j
```

within numerical tolerance.

Persist enough information so a later viewer can answer:

> Which propositions caused the model to prefer top-1 disease over top-2 disease?

---

# 10. S2-G cross-disease contribution audit

For each disease `d`, report at least:

## A. Number of nonzero weights

Use audit threshold:

```text
abs(w_dj) >= nonzero_threshold
```

Default:

```text
1e-4
```

Persist:
- total nonzero weights;
- same-disease nonzero weights;
- cross-disease nonzero weights.

## B. Same-disease contribution fraction

For a chosen split, especially final test after model selection:

```text
same_disease_contribution_fraction_d
=
sum_i sum_{j in P_d} |w_dj * s_ij|
/
sum_i sum_j |w_dj * s_ij|
```

Use absolute contribution magnitude.

Report:
- per disease;
- macro mean;
- median;
- min/max.

For S2-L this should be exactly `1.0` by construction.

Add assertions/tests for S2-L.

---

# 11. Metrics

Persist validation and test metrics at minimum:

```text
top_1_accuracy
top_3_accuracy
macro_f1
balanced_accuracy
macro_roc_auc
mrr
cross_entropy
```

Also save:
- confusion matrix;
- per-disease recall;
- per-disease precision;
- per-disease F1;
- per-disease ROC-AUC where defined.

Do not use thresholded sigmoid metrics for primary Stage-2 evaluation.

---

# 12. Confidence intervals

Add a deterministic paired/bootstrap helper for final reporting.

At minimum support:
- Top-1 accuracy 95% bootstrap CI;
- Top-3 accuracy 95% bootstrap CI.

Use seeded bootstrap.

This helper must operate only AFTER a configuration is frozen.

It must not participate in hyperparameter selection.

---

# 13. Output contract

Each learned Stage-2 run should create:

```text
run_config.json
stage2_concept_manifest.json

checkpoints/
    best.pt
    latest.pt

metrics/
    history.csv
    best_validation.json
    val_summary.json
    test_summary.json
    val_per_disease.csv
    test_per_disease.csv
    confusion_matrix_test.npy

diagnostics/
    concept_weights.csv
    disease_sparsity.csv
    same_disease_contribution_fraction.csv
    contribution_reconstruction_check.json

predictions/
    val_best.npz
    test_best.npz

contributions/
    val_contributions.npz
    test_contributions.npz
```

The NPZ contribution artifact should contain enough for later HTML viewer work:

```text
case_id
disease_id
proposition_id

logits
probabilities

top1_index
top2_index
top1_logit
top2_logit

contribution_top1
contribution_top2
delta_contribution

bias
```

Avoid storing a gigantic `[N, D, P]` tensor unless necessary.

Prefer compact storage for:
- top-1 contributions;
- top-2 contributions;
- top1-top2 delta.

If a full tensor is useful for offline audit and still reasonably sized, it may be stored separately.

---

# 14. Stage-1 input integrity

Stage-2 must treat Stage-1 score artifacts as immutable.

Persist:
- SHA-256 of train/val/test score NPZ files;
- Stage-1 proposition order;
- case order;
- routing mode;
- proposition metadata hash;
- git commit of current Stage-2 run.

Fail loudly if:
- proposition orders differ across splits;
- disease orders differ across splits;
- routing masks differ across splits;
- required score keys are missing.

---

# 15. Tests required

Add comprehensive tests under:

```text
proposition/tests/
```

At minimum:

## S2-L tests
- disease uses only its own propositions;
- polarity `+1` -> `s`;
- polarity `-1` -> `1-s`;
- all effective weights are nonnegative;
- logit reconstructs exactly from bias + contributions;
- same-disease contribution fraction is exactly 1.

## S2-G tests
- uses original scores without polarity flip;
- signed weights supported;
- negative contribution can reduce disease logit;
- L1 penalty computed correctly;
- L2 penalty computed correctly;
- audit `nonzero_threshold` works.

## Routing tests
For A/C/B verify exact selected scalar source:
- image,
- text,
- fused.

## Diagnosis tests
- CrossEntropy 47-class/single-label contract;
- Top-1 / Top-3 correct;
- softmax probabilities sum to 1.

## Explanation tests
- top1/top2 identified correctly;
- `delta_contribution` reconstructs top1-top2 margin;
- bias difference included correctly.

## Selection tests
- practical tie = within 0.02 Top-1;
- local preferred over global when practically tied;
- stronger regularization preferred within same architecture;
- lower val CE resolves remaining tie;
- selector never reads test metrics.

## Reference replication gate
Test the helper itself on synthetic data.

A real notebook run will verify ~60.3% val / ~56.8% test against the official frozen scores.

## Regression
All previous Stage-1 / Stage-2 tests must continue to pass.

Run:

```bash
python -m pytest -q proposition/tests
```

---

# 16. Code organization

Suggested:

```text
proposition/
    models/
        stage2_concept_head.py

    engine/
        stage2_concept_trainer.py
        stage2_concept_metrics.py
        stage2_concept_audit.py

    cli/
        stage2_concept.py
        stage2_concept_reference.py

    tests/
        test_stage2_concept_head.py
        test_stage2_concept_trainer.py
        test_stage2_concept_audit.py
        test_stage2_concept_selection.py
```

Reuse existing Stage-2 score loading/routing code instead of duplicating it.

Do not delete the old Stage-2 implementation yet. Keep it available for historical comparison / regression.

---

# 17. Do NOT implement yet

Do not implement:
- HTML viewer;
- Stage-1 retraining;
- new weak labels;
- new teacher logic;
- disease-name masking changes;
- latent hidden embedding diagnosis path;
- MLP diagnosis head;
- attention-based disease head.

This task is strictly:

> learned linear concept heads over frozen scalar proposition evidence.

---

# 18. No-hardcode rule

New repository code must contain no literal project-specific runtime path such as:

```text
/content/
/content/drive/
MyDrive/
Cao học/
Experiments/K34/
WeakSup/
```

Paths must come from CLI/config.

---

# 19. Delivery

After implementation:

1. commit to branch `WS`;
2. return commit SHA;
3. list all modified/added files;
4. provide full pytest result;
5. confirm no hard-coded paths;
6. confirm old Stage-2 code is preserved;
7. confirm no hidden/latent representation is consumed by the new diagnosis head;
8. do NOT create the experiment notebook.

ChatGPT will review the commit first, then create the Stage-2 experiment notebook.
