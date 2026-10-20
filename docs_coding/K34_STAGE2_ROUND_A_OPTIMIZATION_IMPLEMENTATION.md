# K34 — Round A Stage-2 Optimization Implementation Request

Repository: `trisle0805/RAD-colab`
Branch: `WS`

Current reviewed Stage-2 baseline commit:

```text
5eb394a388395f5925f956d4c7afa01add5e0e72
```

## Goal

Exploit more of the diagnostic information already present in the **frozen scalar Stage-1 proposition evidence**, while preserving the hard concept bottleneck.

This is **Round A** only.

Do NOT change Stage 1, weak labels, encoders, proposition heads, disease masking, or teacher behavior.

Do NOT introduce any hidden/latent representation into diagnosis.

The diagnosis model may consume only scalar evidence values originating from:

```text
score_image
score_text
score_fused
```

from the frozen Stage-1 score artifacts.

---

# 0. Scope of this implementation

Implement the first, cheap part of Round A:

1. add a **dual-channel morphology routing** that keeps image and text scalar evidence separate instead of averaging them;
2. add deterministic **5-fold stratified cross-validation over train + val**;
3. add **train+val refit** after configuration selection;
4. add **3-seed linear-head averaging** as a fixed stability/variance-reduction step;
5. preserve exact proposition-level decomposition for the averaged final head;
6. fix the known contribution reconstruction tolerance issue in a principled way.

Do **NOT** implement GAM/piecewise-additive heads in this commit.

GAM is Round A2 and will only be considered if Round A1 does not improve enough. This avoids adding unnecessary model complexity before testing the cheaper hypothesis.

---

# 1. New routing mode: dual morphology channels

Add a new Stage-2 routing mode:

```text
full_dual_morphology
```

Suggested short label in experiment artifacts:

```text
D
```

The current `full_fused` B routing remains unchanged.

## 1.1 Exact routing semantics

Start conceptually from the existing `full_fused` routing.

For an **active morphology proposition with both image and text evidence**:

current B:

```text
feature = (s_img + s_txt) / 2
```

new D:

```text
feature_1 = s_img
feature_2 = s_txt
```

For all other active propositions:

- text-only -> one text scalar;
- image-only -> one image scalar;
- any non-morphology proposition with both modalities, if such a case exists -> retain the existing `full_fused` rule unless explicitly tested otherwise.

The intended dimensionality is approximately:

```text
307 original proposition concepts
+ one extra modality channel for each morphology proposition with both modalities
≈ 420 scalar features
```

Do not hard-code `113` or `420`; derive them from metadata.

---

# 2. Expanded concept-feature metadata

The current code assumes one feature per proposition.

The new routing requires an explicit expanded feature contract.

Add a dataclass such as:

```python
@dataclass(frozen=True)
class Stage2ConceptFeature:
    feature_id: str
    proposition_id: str
    disease_id: str
    category: str
    polarity: int
    source: str
```

Suggested feature IDs:

```text
<proposition_id>::image
<proposition_id>::text
<proposition_id>::fused
```

For existing A/C/B routes, backwards-compatible IDs are fine, but the expanded routing must be unambiguous.

Persist mappings:

```text
feature_id
base_proposition_id
disease_id
category
polarity
source
```

The Stage-2 head should operate over generic **concept features**, not assume feature count == proposition count.

---

# 3. S2-L and S2-G behavior under dual routing

The existing scientific semantics must remain intact.

## S2-L

Each concept feature inherits:
- disease ownership from its base proposition;
- polarity from its base proposition.

For local head:

```text
positive-polarity feature:
    s_tilde = s

negative-polarity feature:
    s_tilde = 1 - s
```

Weights remain nonnegative:

```text
w >= 0
```

Both `P::image` and `P::text` are allowed if P belongs to disease d.

## S2-G

Use raw scalar feature values:

```text
s in [0,1]
```

Weights remain signed:

```text
w in R
```

No polarity flip.

Diagnosis still satisfies:

```text
logit_d = bias_d + sum_k w_dk * feature_ik
```

where `k` indexes expanded concept features.

No hidden representation is permitted.

---

# 4. Explanation aggregation

Dual channels must remain fully interpretable.

Persist channel-level contribution:

```text
c(feature_k) = weight_k * scalar_feature_k
```

Also provide an aggregation helper by base proposition:

```text
c(proposition_j)
=
sum of contributions of all feature channels belonging to proposition_j
```

Example:

```text
P001::image contribution = +0.40
P001::text  contribution = +0.27

P001 proposition contribution = +0.67
```

The top1-vs-top2 margin decomposition must also work:
- at channel level;
- optionally aggregated by proposition.

Exact reconstruction must remain valid.

---

# 5. Contribution audit numerical tolerance

The current repository audit uses an overly strict absolute tolerance of `1e-6`.

Observed float32 accumulation error in the real Stage-2 run was around:

```text
~7e-6
```

Fix the audit rather than relying on notebook patching.

Required behavior:

- recompute reconstruction in float64 where practical;
- use a documented default audit tolerance:

```text
1e-4
```

or a dtype-aware equivalent no stricter than needed for float32 accumulation over hundreds of concepts.

Persist:

```text
verification_tolerance
max_abs_logit_error
max_abs_margin_error
passes
```

Do not weaken the mathematical identity; this is only a numerical tolerance fix.

Add regression tests.

---

# 6. Deterministic 5-fold CV over train + val

Add a CV utility/CLI.

Suggested entry point:

```bash
python -m proposition.cli.stage2_concept_cv
```

This CLI must accept:

```text
--train_scores
--val_scores

--train_csv
--val_csv

--active_csv
--output_dir

--device
--seed
--folds

--candidate_config_json
```

Do NOT accept or read test files.

Default:

```text
folds = 5
```

## 6.1 Pool construction

Combine the original:

```text
train + val
```

only after validating:
- same proposition order;
- same trainable/active masks;
- disjoint case IDs;
- same disease order.

The pooled set is expected to contain ~1980 cases.

## 6.2 Fold construction

Use deterministic stratified folds by the 47-class disease label.

Suggested:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=seed,
)
```

Persist an immutable fold assignment file:

```text
cv_fold_assignments.csv
```

with:

```text
case_id
disease_id
fold
original_split
```

Hash it and include the hash in the CV manifest.

Each case must appear:
- in exactly one CV validation fold;
- in four CV training folds.

No case may be duplicated or omitted.

---

# 7. Candidate configuration contract

The CV CLI should consume an explicit candidate JSON rather than hard-code experiment choices.

Example:

```json
[
  {
    "run_id": "G-B-l1-1e-4",
    "head_mode": "global",
    "routing_mode": "full_fused",
    "regularization": "l1",
    "lambda_reg": 0.0001,
    "learning_rate": 0.01,
    "batch_size": 256,
    "max_epochs": 300,
    "patience": 40
  },
  {
    "run_id": "G-D-l1-1e-4",
    "head_mode": "global",
    "routing_mode": "full_dual_morphology",
    "regularization": "l1",
    "lambda_reg": 0.0001,
    "learning_rate": 0.01,
    "batch_size": 256,
    "max_epochs": 300,
    "patience": 40
  }
]
```

The notebook will keep the candidate set deliberately small.

Repository code must not hard-code a hyperparameter grid.

---

# 8. CV training and metrics

For every:

```text
candidate × fold
```

train on 4 folds and validate on the remaining fold.

Use the existing:
- 47-class CrossEntropy;
- validation Top-1 primary checkpoint selection;
- validation CE tie-break;
- deterministic seed / minibatch shuffling;
- scalar-only concept head.

Persist per-fold:

```text
candidate_id
fold
best_epoch

top_1_accuracy
top_3_accuracy
macro_f1
balanced_accuracy
macro_roc_auc
mrr
cross_entropy
```

Also persist:
- fold history;
- best checkpoint or at minimum final effective weights/bias required for audit;
- case IDs used in fold train/val.

Aggregate candidate metrics:

```text
mean
std
min
max
```

for all major metrics.

---

# 9. CV-based configuration selection

Add a deterministic helper such as:

```python
select_stage2_cv_configuration(...)
```

It must consume **CV metrics only**.

It must reject any structure containing test metrics.

Primary metric:

```text
mean CV Top-1
```

Practical-tie tolerance:

```text
0.02 absolute mean Top-1
```

Among candidates within 0.02 of the best:

1. prefer the simpler routing:
   - `full_fused` B before `full_dual_morphology` D;
2. prefer Local before Global when otherwise comparable;
3. within same architecture/routing, prefer stronger regularization;
4. final tie-break: lower mean CV cross-entropy.

Persist:
- best mean Top-1;
- tied candidates;
- selected candidate;
- reason.

Do not access test.

---

# 10. Round-A comparison policy

The notebook will primarily compare:

```text
current B = full_fused
new D     = full_dual_morphology
```

using 5-fold CV on train+val.

A/C may remain available for ablation but are not required as primary optimization candidates.

The new D routing should only replace B as the preferred representation if CV evidence supports it under the practical-tie policy.

---

# 11. Train + val refit after selection

Add a separate refit utility/CLI.

Suggested:

```bash
python -m proposition.cli.stage2_concept_refit
```

Inputs:

```text
--train_scores
--val_scores

--train_csv
--val_csv

--active_csv

--selected_config_json
--cv_summary_json

--output_dir
--device

--seeds
```

No test path is allowed in this refit CLI.

## 11.1 Refit epochs

After CV selection, derive a fixed epoch count from the selected candidate's 5 folds.

Use:

```text
refit_epochs = rounded median(best_epoch across the 5 folds)
```

Persist:
- fold best epochs;
- median;
- final refit epoch count.

Do not use the held-out test to determine epochs.

No early stopping during full train+val refit because no validation set remains.

---

# 12. Multi-seed refit / averaging

For the selected configuration, refit on all train+val using exactly three deterministic seeds:

```text
42
43
44
```

unless supplied explicitly by CLI.

Each seed:
- same selected architecture;
- same routing;
- same regularization/lambda;
- same fixed refit epoch count;
- different deterministic initialization/shuffle seed.

Persist per seed:
- effective weights;
- bias;
- training loss history;
- final training loss.

## 12.1 Averaged final linear head

Construct:

```text
W_bar = mean(W_seed)
b_bar = mean(b_seed)
```

This averaged head is the official Round-A ensemble candidate.

Because all components are linear:

```text
mean(logit_seed)
=
b_bar + W_bar * concept_scores
```

so the ensemble remains an exact linear concept head.

For Local:
- average effective nonnegative weights;
- resulting averaged weights must remain nonnegative.

For Global:
- signed averaged weights are allowed.

Persist an explicit averaged-head artifact:

```text
averaged_concept_head.npz
```

containing at least:

```text
feature_id
base_proposition_id
disease_id
weights
bias
seeds
routing_mode
head_mode
regularization
lambda_reg
refit_epochs
```

---

# 13. Multi-seed stability audit

Persist pairwise seed stability:

- Top-1 agreement if a non-test evaluation split is available from CV out-of-fold predictions;
- Pearson/Spearman correlation of flattened effective weights;
- mean/max absolute weight difference;
- bias differences.

Also report averaged-head difference from individual seeds.

This is a stability audit, not a source of hidden features.

---

# 14. Out-of-fold predictions

For each selected CV candidate, generate complete **out-of-fold (OOF)** predictions over all train+val cases.

Each pooled case must receive prediction exactly once from the fold where it was validation.

Persist:

```text
oof_predictions.npz
```

with:

```text
case_id
disease_id
fold
targets
logits
probabilities
```

OOF metrics provide the main Round-A model-selection estimate.

If multi-seed CV is added later, keep it separate; initial CV may use the base seed only.

---

# 15. Final held-out test evaluation must remain separate

Do NOT let:
- CV CLI;
- refit CLI;
- averaging utility

accept or read test files.

Add a separate explicit final evaluation CLI/helper:

```bash
python -m proposition.cli.stage2_concept_evaluate
```

It loads:
- a frozen refit/averaged concept head;
- the held-out test score artifact;
- test labels;
- active proposition metadata.

It must not modify weights.

It exports:
- Top-1;
- Top-3;
- macro-F1;
- balanced accuracy;
- macro ROC-AUC;
- MRR;
- cross-entropy;
- bootstrap 95% CI;
- per-disease metrics;
- confusion matrix;
- contribution artifacts;
- same-disease contribution fraction;
- exact top1/top2 margin decomposition.

The experiment notebook will call this only AFTER:
- CV selection is frozen;
- refit epochs are frozen;
- seed list is frozen;
- averaged head is written and hashed.

---

# 16. Dual-channel viewer artifacts

For D routing, contribution files must preserve:

```text
feature_id
base_proposition_id
source
```

The later viewer must be able to show:

```text
Proposition P:
    image evidence
    text evidence
    image contribution
    text contribution
    total proposition contribution
```

Top1-vs-top2 delta must also be aggregatable by proposition.

Add utility functions for channel-level -> proposition-level aggregation.

---

# 17. Input scale

Production heads must continue using raw scalar scores:

```text
[0,1]
```

Do NOT standardize B or D inputs in learned production heads.

Standardization remains allowed only in historical sklearn reference/diagnostic probes.

---

# 18. Existing Stage-2 baseline must remain intact

Do not delete or silently change:

```text
image_morphology
full_image_morphology_text_other
full_fused
```

Existing Stage-2 tests must still pass.

The previous final S2-G/B result remains a historical baseline.

---

# 19. Tests required

Add comprehensive tests.

## Dual routing
- morphology with both modalities expands into exactly image + text channels;
- no hard-coded feature count;
- text-only remains one text channel;
- image-only remains one image channel;
- non-morph both follows documented B behavior;
- feature IDs are unique and stable;
- feature -> base proposition mapping is correct;
- disease/polarity inheritance is correct.

## Local/global heads with expanded features
- Local dual channels remain disease-local;
- Local polarity flip works independently on image/text channels;
- Local effective weights nonnegative;
- Global uses raw scalar channels;
- signed global weights work.

## Contribution aggregation
- channel contributions reconstruct logits;
- proposition-aggregated contributions sum to channel-level total;
- top1/top2 delta reconstructs margin;
- float64/tolerance audit passes with `1e-4`.

## CV
- train+val case IDs disjoint before pooling;
- deterministic 5-fold stratification;
- every case appears in validation exactly once;
- same seed gives identical folds;
- different seed can change folds;
- CV code has no test input;
- test keys/paths are rejected by CV selector if provided.

## CV selection
- mean CV Top-1 primary;
- 0.02 practical tie;
- B preferred over D when tied;
- Local preferred over Global when otherwise tied;
- stronger regularization preferred;
- lower mean CV CE final tie-break;
- test metrics rejected.

## Refit
- refit epochs = rounded median selected-fold best epochs;
- all train+val cases used;
- no test access;
- fixed epoch count;
- seeds deterministic.

## Averaged head
- averaged weight/bias equal arithmetic mean;
- averaged-head logits equal mean of seed logits;
- contribution decomposition exact;
- Local averaged weights remain nonnegative.

## Final evaluation
- loads frozen averaged head without optimizer/training;
- cannot mutate weights;
- produces required metrics/CI/contribution audit.

## Regression
Run:

```bash
python -m pytest -q proposition/tests
```

All previous Stage-1 and Stage-2 tests must continue to pass.

---

# 20. No-hardcode rule

New repository code must not contain project-specific runtime paths:

```text
/content/
/content/drive/
MyDrive/
Cao học/
Experiments/K34/
WeakSup/
```

All paths come from CLI/config.

---

# 21. Suggested code organization

Possible additions:

```text
proposition/data/
    stage2_concept_features.py

proposition/engine/
    stage2_concept_cv.py
    stage2_concept_refit.py
    stage2_concept_ensemble.py

proposition/cli/
    stage2_concept_cv.py
    stage2_concept_refit.py
    stage2_concept_evaluate.py

proposition/tests/
    test_stage2_dual_routing.py
    test_stage2_concept_cv.py
    test_stage2_concept_refit.py
    test_stage2_concept_ensemble.py
```

Reuse the existing concept head/metrics/audit implementation where practical.

Avoid duplicating Stage-2 score loading.

---

# 22. Do NOT implement in this commit

Do not implement:

- GAM / piecewise additive functions;
- MLP head;
- latent embedding diagnosis;
- Stage-1 fine-tuning;
- BERT LoRA;
- min-label changes;
- image encoder changes;
- new weak labels;
- HTML viewer.

Those are later decisions.

---

# 23. Delivery

After implementation:

1. commit to branch `WS`;
2. return commit SHA;
3. list all added/modified files;
4. give full `pytest` result;
5. confirm no hard-coded path;
6. confirm CV/refit code cannot access test;
7. confirm diagnosis still consumes only scalar evidence;
8. confirm existing B routing and previous Stage-2 implementation remain available;
9. do NOT create the experiment notebook.

ChatGPT will review the commit first, then create the Round-A experiment notebook.
