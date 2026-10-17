"""Disease-agnostic masking for captions consumed by the Stage-1 student."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

DEFAULT_MASK_TOKEN = "[DISEASE]"


def disease_aliases(diseases: Mapping[str, Iterable[str]] | Iterable[str]) -> tuple[str, ...]:
    """Return a de-duplicated, globally applicable set of non-empty aliases.

    A mapping may associate canonical disease IDs with their aliases. Its keys
    are included as aliases too, so callers cannot accidentally leave canonical
    diagnostic names unmasked.
    """
    candidates: list[str] = []
    if isinstance(diseases, Mapping):
        for disease, aliases in diseases.items():
            candidates.append(str(disease))
            candidates.extend(str(alias) for alias in aliases)
    else:
        candidates.extend(str(alias) for alias in diseases)

    unique = {candidate.strip() for candidate in candidates if candidate and candidate.strip()}
    return tuple(sorted(unique, key=lambda alias: (-len(alias), alias.casefold())))


class DiseaseAliasMasker:
    """Masks every configured disease alias, irrespective of a case's label."""

    def __init__(
        self,
        aliases: Mapping[str, Iterable[str]] | Iterable[str],
        *,
        mask_token: str = DEFAULT_MASK_TOKEN,
    ) -> None:
        if not mask_token:
            raise ValueError("mask_token must not be empty")
        self.aliases = disease_aliases(aliases)
        self.mask_token = mask_token
        self._pattern = _compile_alias_pattern(self.aliases)

    def mask(self, caption: str) -> str:
        if not isinstance(caption, str):
            raise TypeError("caption must be a string")
        if self._pattern is None:
            return caption
        return self._pattern.sub(self.mask_token, caption)

    def __call__(self, caption: str) -> str:
        return self.mask(caption)


def mask_disease_aliases(
    caption: str,
    aliases: Mapping[str, Iterable[str]] | Iterable[str],
    *,
    mask_token: str = DEFAULT_MASK_TOKEN,
) -> str:
    """Mask global disease aliases in ``caption`` without using a case label."""
    return DiseaseAliasMasker(aliases, mask_token=mask_token).mask(caption)


def _compile_alias_pattern(aliases: tuple[str, ...]) -> re.Pattern[str] | None:
    if not aliases:
        return None
    # Alphanumeric aliases use word boundaries; punctuation aliases still need
    # protection against matching inside a larger word on either side.
    alternatives = "|".join(re.escape(alias) for alias in aliases)
    return re.compile(r"(?<!\w)(?:" + alternatives + r")(?!\w)", re.IGNORECASE)
