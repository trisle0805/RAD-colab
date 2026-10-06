# Proposition extension

This package contains the proposition-based extension built on top of the RAD baseline.

The original RAD implementation remains in the repository's existing `dataset`, `engine`, `factory`, and `models` packages. New proposition-specific data loading, models, losses, training utilities, artifacts, and tests are isolated here.

The dedicated experiment entry point will be `main_proposition.py`; `main_rad.py` remains the baseline entry point.
