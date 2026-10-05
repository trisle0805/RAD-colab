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
| `top1_accuracy` | Fraction of samples where the `argmax` class is correct. |
| `top3_accuracy` | Fraction whose true class occurs among the three highest-scoring classes. |
| `balanced_accuracy` | Mean per-class recall using `argmax` predictions. |
| `macro_auc` | Mean one-vs-rest ROC-AUC score. |
| `mAP` | Mean average precision. |
| `rad_legacy_metrics` | Original RAD threshold-based metrics retained for reference only. |

Use `macro_f1` as the principal selection metric because each class is weighted equally. `top1_accuracy`, `top3_accuracy`, and `balanced_accuracy` are complementary single-label metrics.

The legacy `label_mean_accuracy` and `guideline_mean_accuracy` fields are not suitable as headline SkinCAP scores: their binary per-label calculation is dominated by negative labels in a one-hot 47-class problem.
