# K34 — Round B1 Stage-1 BioClinicalBERT Adaptation Implementation Request

Repository: `trisle0805/RAD-colab`  
Branch: `WS`

Current Round-A head implementation commit:

```text
89c7246cf15af872468b9a4587a5d1f637bcb374
```

## Goal

Improve the **Stage-1 text proposition evidence** by lightly adapting BioClinicalBERT, while preserving the thesis's hard proposition bottleneck and the already validated image branch.

This is **Round B1** only.

The diagnosis path must remain:

```text
patient input
    ↓
Stage-1 proposition evidence
    ↓
scalar proposition scores
    ↓
Stage-2 concept head
    ↓
diagnosis
```

Do NOT allow Stage 2 to consume BioClinicalBERT hidden states or any other latent representation.

Do NOT use disease supervision during Stage 1.

---

# 1. Scientific contract

The Stage-1 text target semantics remain EXACTLY:

```text
PRESENT -> 1
ABSENT  -> 0
UNKNOWN -> 0
```

Meaning:

> Does the masked patient caption provide textual support for proposition j?

Loss remains binary cross-entropy with logits.

The current TARGETED weak-label teacher is intentionally unchanged.

Do NOT:
- regenerate weak labels;
- alter TARGETED teacher behavior;
- change `min_labels`;
- change disease-name masking;
- change the image weak labels;
- introduce disease classification loss;
- jointly train Stage 1 and Stage 2.

For Round B1, keep the same active proposition set used by the successful text-support run.

---

# 2. Initialization source

Round B1 starts from the **successful Stage-1 text-support checkpoint**, not from a fresh BioClinicalBERT model.

The notebook will provide:

```text
--init_checkpoint
```

pointing to the best checkpoint from the validated text-support run:

```text
stage1_text_support/text_support_s1e49_56da5613
```

Repository code must not hard-code that path.

Load:
- image branch weights;
- text cross-attention weights;
- text head weights;
- BioClinicalBERT weights from that checkpoint/model state.

Start a NEW experiment:
- fresh optimizer;
- fresh history;
- epoch starts from 1;
- no restored optimizer state;
- no restored early-stopping counter;
- no restored RNG state.

Validate proposition order and weak-supervision artifact hashes before training.

---

# 3. Adaptation modes

Implement a generic text-encoder adaptation configuration.

Required modes:

```text
frozen
last1
last2
```

Meaning:

## `frozen`

Existing validated behavior:

```text
BioClinicalBERT frozen
text_cross_attention trainable
text_head trainable
```

This mode is mainly for regression/contract testing. The notebook may reuse the already-completed frozen baseline instead of retraining it.

## `last1`

Train:

```text
last BioClinicalBERT encoder layer
text_cross_attention
text_head
```

Everything else frozen.

## `last2`

Train:

```text
last 2 BioClinicalBERT encoder layers
text_cross_attention
text_head
```

Everything else frozen.

Do NOT implement full-BERT fine-tuning in Round B1.

Do NOT implement LoRA in this commit. LoRA may be considered later only if last1/last2 are insufficient.

---

# 4. IMPORTANT: fixed proposition query anchor

This is a non-negotiable Round-B scientific constraint.

The current model uses BioClinicalBERT both to represent:
- the patient caption memory;
- canonical proposition descriptions used as queries.

If the query representation is allowed to move while adapting BERT, the image proposition scores would also change even though the image branch itself is frozen.

Therefore Round B1 must use a **fixed proposition query anchor**.

## 4.1 How to create it

Immediately AFTER loading the validated Stage-1 initialization checkpoint, but BEFORE unfreezing any BioClinicalBERT layer:

1. put the text encoder in deterministic `eval()` mode;
2. encode all active `canonicalDescription` proposition strings;
3. extract the same pooled query representation used by the current model;
4. detach it;
5. store it as the immutable query anchor.

Conceptually:

```python
query_anchor = encode_canonical_descriptions_with_initial_frozen_bert()
query_anchor = query_anchor.detach()
```

## 4.2 Query anchor must never change during adaptation

After training starts:

- never recompute the normal proposition query cache from adapted BioClinicalBERT;
- never backpropagate through query construction;
- never update the query anchor;
- normal forward passes must use the frozen anchor.

This preserves:
- canonical proposition semantics;
- exact comparability with the previous Stage-1 image branch;
- the meaning of query-intervention audits.

## 4.3 Persist it

Current `query_cache` is non-persistent.

Do NOT break backwards compatibility with old Stage-1 checkpoints.

Instead, explicitly persist the immutable query anchor in the new Round-B checkpoint/artifact, for example:

```text
query_anchor
query_anchor_sha256
```

The final export/evaluation CLI must restore the exact saved query anchor.

Fail loudly if:
- the anchor is missing from a Round-B checkpoint;
- shape/proposition order differs;
- anchor hash differs after restoration.

---

# 5. Stage1EvidenceModel training scope

Extend the model cleanly without breaking existing modes.

Existing modes must continue working:

```text
full_stage1
text_only
```

Add a Round-B mode such as:

```text
text_adapt
```

with an explicit adaptation setting:

```text
bert_adaptation = frozen | last1 | last2
```

Suggested API names are flexible, but behavior must be explicit and testable.

---

# 6. Exact gradient behavior

The current `Stage1EvidenceModel.forward()` evaluates text encoding inside `torch.no_grad()`.

That is correct for the frozen baseline but MUST change for adapted caption encoding.

Required behavior:

## Image

Always frozen:

```python
with torch.no_grad():
    image_memory = ...
```

No image gradient in any Round-B mode.

## Text, frozen mode

Equivalent to current behavior:

```python
with torch.no_grad():
    text_memory = ...
```

## Text, `last1` / `last2`

Caption text memory MUST be computed with autograd enabled:

```python
text_memory = encode_caption_with_bert(...)
```

so gradients reach only the explicitly trainable BioClinicalBERT layers.

The fixed proposition query anchor remains detached and receives no gradient.

---

# 7. Exact BioClinicalBERT layers to train

The current wrapper exposes:

```text
text_encoder.bert_model
```

and HuggingFace BERT encoder layers through approximately:

```text
text_encoder.bert_model.encoder.layer
```

Do not depend on the wrapper's historical `freeze_layers` argument for this experiment.

Implement an explicit helper that:

1. freezes the entire text encoder first;
2. locates the BERT encoder layer stack;
3. unfreezes only the requested last `N` layers.

For:

```text
last1 -> encoder.layer[-1]
last2 -> encoder.layer[-2:]
```

Keep frozen:
- token embeddings;
- positional embeddings;
- earlier BERT layers;
- pooler if present;
- wrapper `mlp_embed`;
- wrapper `logit_scale`.

The Stage-1 path currently uses `bert_model` sequence output directly; do not silently start using `mlp_embed`.

Fail loudly if the expected BERT layer stack cannot be found.

---

# 8. Module train/eval modes

`requires_grad=False` is not enough.

During adapted training:

## Always eval/frozen

```text
image_encoder
image_projection
image_cross_attention
image_head

all frozen BioClinicalBERT submodules
```

## Train

```text
text_cross_attention
text_head
selected last BioClinicalBERT layer(s)
```

The selected BERT layer(s) may use their normal training-time dropout.

Frozen lower BERT layers should remain `eval()` so their dropout is not stochastic.

Add assertions that the module modes match the requested adaptation scope after every `model.train()` call.

---

# 9. Optimizer parameter groups

Do NOT use one learning rate blindly for all trainable parameters.

Round-B optimizer should have two parameter groups:

## Text evidence head

```text
text_cross_attention
text_head
```

Uses:

```text
head_lr
```

## Adapted BioClinicalBERT layers

Only selected BERT layer(s).

Uses:

```text
bert_lr
```

Notebook will choose the experiment values.

Repository code must expose both.

Suggested defaults:

```text
head_lr = 1e-4
bert_lr = 1e-5
weight_decay = 1e-4
```

Do not hard-code notebook experiment choices beyond safe CLI defaults.

Add an optimizer-contract assertion proving that optimizer parameter IDs equal EXACTLY:

```text
text_cross_attention
+ text_head
+ explicitly selected BERT parameters
```

No other parameter may enter the optimizer.

---

# 10. Gradient clipping

Add optional gradient clipping for adapted text training.

CLI/config:

```text
--max_grad_norm
```

Suggested default:

```text
1.0
```

Apply after backward and before optimizer step.

Persist actual value in run manifest.

Frozen baseline mode may use the same code path safely.

---

# 11. Training loss

Round B remains a TEXT-ONLY experiment.

Optimization objective:

```text
loss = textual-support BCE only
```

under:

```text
PRESENT -> 1
ABSENT/UNKNOWN -> 0
```

Do NOT add:
- image BCE;
- disease CE;
- contrastive loss;
- hidden-state regularizers;
- Stage-2 loss.

Image predictions may be computed for freeze verification only.

---

# 12. Checkpoint selection

Select the best Round-B checkpoint using validation text evidence only.

Primary:

```text
validation text macro ROC-AUC
```

Strict eligibility:

```text
min effective positive = 5
min effective negative = 5
```

Tie-break:

```text
lower validation text BCE
```

Persist:

```text
best_epoch
best_validation_text_macro_auc
best_validation_text_bce
```

Do not use diagnosis accuracy to select the Stage-1 adaptation checkpoint.

Do not use held-out test metrics for selection.

---

# 13. Train/validation CLI

Add a new CLI:

```bash
python -m proposition.cli.text_adapt
```

Required arguments:

```text
--config
--init_checkpoint

--labels_npz
--active_masks_npz
--active_csv
--metadata_json

--train_csv
--val_csv
--image_root
--disease_aliases_json

--output_dir
--device

--adaptation
    frozen
    last1
    last2

--epochs
--patience
--seed

--head_lr
--bert_lr
--weight_decay
--max_grad_norm

--selection_min_present
--selection_min_absent
```

IMPORTANT:

This training CLI must NOT accept or read:
- test scores;
- test CSV;
- Stage-2 diagnosis targets beyond what the existing data loader may contain but must ignore;
- diagnosis-loss configuration.

It trains and selects using train + val only.

---

# 14. Final export/evaluation CLI

Add a separate explicit CLI:

```bash
python -m proposition.cli.text_adapt_export
```

It loads:
- the selected Round-B checkpoint;
- the exact saved query anchor;
- weak-supervision artifacts;
- data splits supplied by CLI.

It must NOT train or modify weights.

Required support:

```text
--checkpoint
--labels_npz
--active_masks_npz
--active_csv
--metadata_json

--train_csv
--val_csv
--test_csv

--image_root
--disease_aliases_json

--output_dir
--device
--seed
```

This is the only Round-B path that may read test.

The notebook will invoke it only after an adaptation configuration and checkpoint are frozen.

---

# 15. Stage-2 compatible score exports

Final export must produce:

```text
predictions/
    scores_train_best.npz
    scores_val_best.npz
    scores_test_best.npz
```

with the SAME contract consumed by Round-A Stage 2:

```text
case_id
proposition_id

score_image
score_text
score_fused

image_trainable
text_trainable
active_scoring
```

`score_fused` retains the existing Stage-1 fusion rule for compatibility.

Round-A `full_dual_morphology` will consume the separate `score_image` and `score_text` fields.

Do not change the NPZ key contract.

---

# 16. Image-branch invariance is mandatory

Because Round B adapts only caption BioClinicalBERT and keeps the proposition query anchor fixed, the image branch must remain identical to the validated initialization checkpoint.

Before adaptation, record deterministic image scores on a fixed validation batch.

After restoring the selected best checkpoint, recompute them.

Required:

```text
max_abs_image_score_diff <= numerical tolerance
```

Expected should be essentially zero.

Also hash before/after:

```text
image_encoder
image_projection
image_cross_attention
image_head
```

All must be identical.

Fail loudly if image evidence changes.

This is a core scientific control.

---

# 17. Query-anchor invariance

Persist:

```text
query_anchor_sha256_before
query_anchor_sha256_after
query_anchor_equal = true
```

Also verify exact tensor equality after:
- training;
- restoring best checkpoint;
- final export.

Fail if the normal anchor changes.

---

# 18. Frozen vs adapted BERT audit

Create an explicit parameter/module audit.

Persist at least:

```text
adaptation_mode
trainable_parameter_count
trainable_bert_parameter_count

trainable_parameter_names
frozen_parameter_names_or_summary

pre_hash
post_hash
changed
allowed_to_change
```

At minimum hash separately:

```text
bert_embeddings
each BERT encoder layer
bert_pooler_if_present
text_cross_attention
text_head
```

Rules:

## frozen

Only:

```text
text_cross_attention
text_head
```

may change.

## last1

Only:

```text
last BERT layer
text_cross_attention
text_head
```

may change.

## last2

Only:

```text
last two BERT layers
text_cross_attention
text_head
```

may change.

Fail if a frozen BERT layer changes.

For an allowed adapted layer, if its hash does NOT change after non-trivial training, report/fail because adaptation may not actually be receiving gradients.

---

# 19. Text evidence metrics

Persist train/val and final-export test metrics:

```text
macro ROC-AUC
macro PR-AUC
n eligible strict 5/5
mean score
std score
quantiles
mean positive score
mean negative score
fraction >= 0.5
```

These remain:

> agreement with effective weak textual-evidence targets,

not clinical ground truth.

---

# 20. Query-intervention support

The three scientific audits must remain possible for adapted text evidence.

The export/evaluation implementation must support deterministic query interventions using the **saved fixed query anchor**.

At minimum generate on validation:

```text
diagnostics/query_controls_val.npz
```

with:

```text
case_id
proposition_id

baseline_score_text
query_swap_score_text
query_zero_score_text
query_mean_score_text
query_random_score_text
```

Use a deterministic seed for:
- swap mapping;
- random queries.

All controls must be evaluated against the same:

```text
caption_support_binary
```

effective targets.

Do not regenerate queries from the adapted BERT encoder.

---

# 21. Scientific audit readiness

The Round-B notebook will later compute the same three audits used to validate the current text branch:

1. disease-only prior baseline;
2. within-disease stratified AUC;
3. query intervention degradation.

Repository code does not need to decide PASS/FAIL scientifically, but it must export enough artifacts to run those audits without retraining.

The final score NPZ plus `query_controls_val.npz` must be sufficient.

---

# 22. Output contract — adaptation training

A training run should create at least:

```text
run_manifest.json

checkpoints/
    best.pt
    latest.pt

metrics/
    history.csv
    best_validation.json
    text_adapt_summary.json

supervision/
    label_distribution.csv

diagnostics/
    query_anchor.npz
    query_anchor_audit.json
    module_parameter_audit.json
    image_branch_freeze_audit.json
```

Do not export held-out test from this training run.

---

# 23. Output contract — final frozen export

Final export should create:

```text
export_manifest.json

predictions/
    scores_train_best.npz
    scores_val_best.npz
    scores_test_best.npz

metrics/
    text_evidence_train.json
    text_evidence_val.json
    text_evidence_test.json

diagnostics/
    query_controls_val.npz
    query_anchor_audit.json
    image_branch_freeze_audit.json
```

Optional compact CSV summaries are welcome.

---

# 24. Run metadata

Persist:

```text
git_commit
adaptation_mode

init_checkpoint_sha256
init_checkpoint_git_commit_if_available

weak_supervision_artifact_hashes

query_anchor_sha256

head_lr
bert_lr
weight_decay
max_grad_norm

seed
best_epoch

trainable_parameter_count
trainable_bert_parameter_count
```

All paths come from CLI/config.

---

# 25. Existing behavior must remain unchanged

Do not break:

```text
proposition.cli.text_retrain
```

or the validated frozen text-support workflow.

Existing:

```text
training_scope="text_only"
```

must retain its exact current meaning:
- frozen BioClinicalBERT;
- fixed image branch;
- train only text cross-attention + text head;
- `caption_support_binary`.

Round-B adaptation should be additive, not a silent semantic change to the baseline.

---

# 26. Tests required

Add comprehensive tests under:

```text
proposition/tests/
```

At minimum:

## Query anchor
- created from initialization encoder before adaptation;
- detached/no gradient;
- unchanged after optimizer steps;
- saved in Round-B checkpoint;
- restored exactly;
- old checkpoints remain loadable by old workflow;
- adapted export fails if anchor is missing/corrupt.

## Gradient flow
For `last1`:
- last BERT layer receives gradient;
- earlier layers receive no gradient;
- text attention/head receive gradient;
- image modules receive no gradient.

For `last2`:
- last two BERT layers receive gradient;
- earlier layers do not.

For `frozen`:
- no BERT layer receives gradient.

## `torch.no_grad()` regression
- adapted caption forward is NOT enclosed by no-grad;
- changing trainable BERT params changes text memory/text score;
- image score remains unchanged because query anchor and image branch are fixed.

## Module modes
- frozen BERT submodules stay eval;
- selected BERT layers are train;
- image modules stay eval;
- text attention/head are train.

## Optimizer contract
- exactly head params + selected BERT params;
- separate head/bert LR groups;
- no frozen params included.

## Objective
- text-only BCE;
- UNKNOWN contributes as effective negative;
- image loss excluded;
- disease target cannot enter loss.

## Selection
- validation text macro ROC-AUC primary;
- lower validation text BCE tie-break;
- test cannot affect checkpoint selection.

## Image invariance
- image weights unchanged;
- image scores unchanged with fixed query anchor.

## BERT hash audit
- only allowed BERT layers change;
- frozen layers do not.

## Export
- Stage-2-compatible train/val/test score NPZ contract;
- proposition and case order preserved;
- exact query anchor restored;
- final exporter has no optimizer/training step.

## Query controls
- deterministic for same seed;
- baseline uses normal fixed anchor;
- swap/zero/mean/random shapes match;
- controls do not mutate the anchor/model.

## Regression
All existing Stage-1 and Stage-2 tests must continue to pass:

```bash
python -m pytest -q proposition/tests
```

---

# 27. No-hardcode rule

New code must not contain project-specific runtime paths:

```text
/content/
/content/drive/
MyDrive/
Cao học/
Experiments/K34/
WeakSup/
```

Paths come from CLI/config.

---

# 28. Suggested organization

Possible additions:

```text
proposition/engine/
    text_encoder_adaptation.py

proposition/cli/
    text_adapt.py
    text_adapt_export.py

proposition/tests/
    test_text_encoder_adaptation.py
    test_text_adapt_cli.py
    test_text_adapt_export.py
```

Minimal changes will also be needed in:

```text
proposition/models/stage1_evidence_model.py
proposition/engine/trainer.py
```

Prefer reuse of existing:
- Stage-1 dataset;
- weak-supervision loading;
- masked BCE;
- metrics;
- score export contract.

Do not duplicate existing pipeline logic unnecessarily.

---

# 29. Do NOT implement in Round B1

Do not implement:
- LoRA;
- full BERT fine-tuning;
- `min_labels` change;
- new weak labels;
- new teacher;
- image encoder replacement;
- GAM;
- MLP disease head;
- hidden-state disease classifier;
- joint Stage-1/Stage-2 optimization;
- HTML viewer.

Round B1 asks one clean question:

> Does adapting only the last 1–2 BioClinicalBERT layers improve proposition-level textual evidence, and therefore downstream diagnosis, while preserving the hard scalar bottleneck?

---

# 30. Delivery

After implementation:

1. commit to branch `WS`;
2. return commit SHA;
3. list all added/modified files;
4. provide full pytest result;
5. confirm no hard-coded paths;
6. confirm old frozen `text_retrain` behavior is preserved;
7. confirm image branch/query anchor invariance;
8. confirm diagnosis supervision is absent from Stage-1 adaptation;
9. confirm Stage-2 score export remains backward compatible;
10. do NOT create the experiment notebook.

ChatGPT will review the commit first.

Only after code review passes will ChatGPT create the Round-B experiment notebook comparing:

```text
frozen baseline
vs
last1
vs
last2
```

and then run the scientific audits before any downstream Stage-2 decision.
