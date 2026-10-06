# Proposition extension

This package contains the proposition-based extension built on top of the RAD baseline.

The original RAD implementation remains in the repository's existing `dataset`, `engine`, `factory`, and `models` packages. New proposition-specific data loading, models, losses, training utilities, artifacts, and tests are isolated here.

The dedicated experiment entry point is `main_proposition.py`; `main_rad.py` remains the unchanged baseline entry point.

## Complete experiment artifacts

A completed `main_proposition.py` run writes the comparable validation history to
`<output_dir>/metrics/metrics_history.csv`, final test metrics to
`<output_dir>/final_metrics.json`, and primary test predictions to
`<output_dir>/predictions/test_best.npz`.

It also performs one extra **test-only** pass from the selected `best.pt` checkpoint
and writes UI-ready explanation data under `<output_dir>/checklist/`:

- `test_proposition_evidence.npz`: disease probabilities, raw support,
	polarity-aware compatibility, pre-calibration disease scores, full proposition
	attention, attention entropy, top attention positions, caption tokens, labels,
	and image paths;
- `proposition_index.json` and `disease_index.json`: stable IDs and evidence
	provenance needed to interpret tensors;
- `test_proposition_evidence.schema.json`: array shapes, memory layout, and
	renderer metadata.

For SkinCAP, image attention occupies the first `image_token_count` memory
positions; caption-token attention occupies the remaining positions. The schema
records the split rather than relying on a hard-coded token count.

## Render attention reports

The renderer consumes saved checklist artifacts only; it does not train or load a
model checkpoint. Run it after a completed experiment:

```text
python proposition/visualization/render_attention.py \
	--evidence <output_dir>/checklist/test_proposition_evidence.npz \
	--bert_model_name <the same ClinicalBERT path or Hugging Face ID> \
	--samples 0 1 2
```

It creates `<output_dir>/checklist/rendered_attention/sample_XXXX/report.html`.
Each report contains the top predicted diseases, the strongest propositions for
each disease, image-patch heatmaps, caption-token highlighting, support,
compatibility, attention entropy, polarity, and evidence provenance. These HTML
reports and the underlying NPZ schema are intended as the data contract for a
future inference UI.
