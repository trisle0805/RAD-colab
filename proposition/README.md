# Stage 1 proposition evidence learning

This package is the isolated implementation of **Stage 1: weakly supervised
proposition evidence learning**. It learns evidence for whether a proposition
concept is present in an image or disease-masked caption. It does **not**
predict diseases, aggregate propositions into disease scores, apply polarity,
or use disease labels in the model or loss.

The RAD baseline remains outside this package. In particular,
[`main_rad.py`](../main_rad.py) is unchanged.

## Current package structure

```text
proposition/
+-- data/       # KB and immutable v3 weak-supervision contracts
+-- models/     # Stage-1 frozen-backbone evidence model (added in Commit 3)
+-- losses/     # Masked weak-supervision loss (added in Commit 4)
+-- engine/     # Trainer, metrics, artifacts, diagnostics (Commits 5–6)
+-- tests/      # Tests only for the current Stage-1 architecture
```

## Stage 1 weak-supervision contracts

The immutable v3 artifacts are loaded through
`proposition.data.weak_supervision`:

- `labels_raw_v3.npz` requires `case_id`, `split`, `proposition_id`,
  `label_text`, `label_img`, and `p_img`;
- `active_masks_v3.npz` requires proposition IDs plus text/image trainability,
  active-scoring flags, and `min_labels`;
- `active_propositions_v3.csv` is reconciled by `proposition_id` and provides
  audit metadata; and
- `metadata_final_v3.json` is retained as the manifest and can optionally
  verify SHA256 digests.

Labels retain `1 = PRESENT`, `0 = ABSENT`, and `-1 = UNKNOWN`. Unknown labels
are never converted to absent labels. Matrix columns are aligned by their
explicit proposition IDs (and case lookup is by `case_id`), rather than by an
assumed file order. Inactive propositions remain unavailable (`N/A`), not zero
scores.