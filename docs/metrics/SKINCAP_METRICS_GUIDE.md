# SkinCAP-47 metrics guide

SkinCAP is a **47-class, single-label** classification task. The class order is the CSV header from column 3 onward (`skincap_47_train.csv`) and must exactly match the 47 `query` values in `guideline/qwen_maxtoken2k_skincap47_4sources.jsonl`.

## Train, validation, and test protocol

- Train with `skincap_47_train.csv`.
- Validate after each epoch with `skincap_47_val.csv`, in stable CSV order. Validation predictions are stored as `predictions/val_epoch_XXX.npz`.
- Select the checkpoint using `max(label_macro_f1, guideline_macro_f1)` on validation. `metrics/metrics_history.csv` records both the epoch candidate (`validation_score`) and the running best (`best_validation_score`, `best_epoch`).
- The best checkpoint is saved as `checkpoints/best.pt`; resumable training state is `checkpoints/latest.pt`.
- Only after training, reload `best.pt` and evaluate `skincap_47_test.csv` once. The prediction artifact is `predictions/test_best.npz`; final results are in `final_metrics.json`.

Per-class fusion weights are fitted only with validation predictions from `best_epoch`, optimizing macro-F1. Those frozen weights are then applied to test predictions; test data is never used to select a checkpoint or tune fusion.

## Metrics

Each stream in `final_metrics.json` (`label`, `guideline`, and `fused`) contains:

| Metric | Definition |
|---|---|
| `macro_f1` | Mean F1 across the 47 classes after selecting one class with `argmax`. |
| `top{k}_accuracy` | For every requested `k` in `--topk` (default `1 2 3`), fraction of samples whose true-label rank is at most `k`. |
| `mrr_at_{k}` | For each requested `k >= 2`, mean reciprocal true-label rank, contributing `1 / rank` only when rank is at most `k`. |
| `macro_mrr_at_{k}` | For each requested `k >= 2`, unweighted mean of per-class `mrr_at_{k}` values across classes represented in the split. |
| `balanced_accuracy` | Mean per-class recall using `argmax` predictions. |
| `macro_auc` | Mean one-vs-rest ROC-AUC score. |
| `mAP` | Mean average precision. |
| `rad_legacy_metrics` | Original RAD threshold-based metrics retained for reference only. |

Ranks use the strict comparison `1 + count(score > true_label_score)`, so scores tied with the true label do not worsen its rank. Metrics are therefore deterministic under the documented competition-rank convention and do not rely on arbitrary tie-breaking at a top-k cutoff. `mrr_at_1` is omitted because it is identical to `top1_accuracy` under this definition.

When the default cutoffs `1 2 3` are used, the implementation verifies the invariant `mrr_at_3 = 0.5 * top1_accuracy + (1 / 6) * top2_accuracy + (1 / 3) * top3_accuracy` to within `1e-9`.

Use `macro_f1` as the principal selection metric because each class is weighted equally. Ranking metrics are recorded for the label and guideline streams in `metrics/metrics_history.csv`, and for label, guideline, and fused streams in `final_metrics.json`; they do not alter checkpoint selection.

The legacy `label_mean_accuracy` and `guideline_mean_accuracy` fields are not suitable as headline SkinCAP scores: their binary per-label calculation is dominated by negative labels in a one-hot 47-class problem.
