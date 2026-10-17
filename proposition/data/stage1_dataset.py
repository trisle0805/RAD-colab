"""SkinCAP dataset adapter for Stage-1 evidence learning."""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from torch.utils.data import Dataset
except ImportError:  # Allows artifact and masking utilities without torch installed.
    class Dataset:  # type: ignore[no-redef]
        pass

if TYPE_CHECKING:
    from PIL import Image

from .caption_masking import DEFAULT_MASK_TOKEN, DiseaseAliasMasker
from .weak_supervision import WeakSupervisionArtifacts

ImageTransform = Callable[[Any], Any]


class Stage1SkinCAPDataset(Dataset):
    """Load SkinCAP rows and attach immutable weak labels by explicit case ID.

    Disease labels are retained solely for diagnostics/evaluation. They are not
    passed to caption masking and are never used to select weak-label rows.
    """

    def __init__(
        self,
        csv_path: str | Path,
        *,
        weak_supervision: WeakSupervisionArtifacts,
        disease_aliases: Mapping[str, Iterable[str]] | Iterable[str],
        image_root: str | Path | None = None,
        split: str | None = None,
        transform: ImageTransform | None = None,
        case_id_column: str = "case_id",
        image_column: str = "image",
        caption_column: str = "text",
        disease_label_columns: Sequence[str] | None = None,
        mask_token: str = DEFAULT_MASK_TOKEN,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root) if image_root is not None else None
        self.transform = transform
        self.case_id_column = case_id_column
        self.image_column = image_column
        self.caption_column = caption_column
        self._mask_caption = DiseaseAliasMasker(disease_aliases, mask_token=mask_token)
        self.weak_supervision = weak_supervision
        self._weak_index = _case_index(weak_supervision)
        rows, fieldnames = _read_csv(self.csv_path)
        _require_columns(fieldnames, (case_id_column, image_column, caption_column), self.csv_path)
        self.disease_label_columns = tuple(disease_label_columns or _default_label_columns(fieldnames))
        _require_columns(fieldnames, self.disease_label_columns, self.csv_path)

        self._records: list[dict[str, Any]] = []
        seen_case_ids: set[str] = set()
        for row_number, row in enumerate(rows, start=2):
            case_id = row[case_id_column].strip()
            if not case_id:
                raise ValueError(f"{self.csv_path}:{row_number}: case_id must not be empty")
            if case_id in seen_case_ids:
                raise ValueError(f"{self.csv_path}: duplicate case_id {case_id!r}")
            seen_case_ids.add(case_id)
            try:
                weak_position = self._weak_index[case_id]
            except KeyError as error:
                raise KeyError(f"{self.csv_path}:{row_number}: no weak labels for case_id {case_id!r}") from error
            artifact_split = str(weak_supervision.splits[weak_position])
            if split is not None and artifact_split != split:
                continue
            self._records.append(
                {
                    "case_id": case_id,
                    "image_path": row[image_column].strip(),
                    "caption": row[caption_column],
                    "split": artifact_split,
                    "disease_label": _disease_label(row, self.disease_label_columns, self.csv_path, row_number),
                    "weak_index": weak_position,
                }
            )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._records[index]
        try:
            from PIL import Image
        except ImportError as error:
            raise ImportError("Stage1SkinCAPDataset requires Pillow to load images") from error
        image = Image.open(self._resolve_image_path(record["image_path"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "case_id": record["case_id"],
            "image": image,
            "caption": self._mask_caption(record["caption"]),
            "split": record["split"],
            "disease_label": record["disease_label"],
        }

    def weak_labels(self, case_ids: Sequence[str]) -> tuple[Any, Any]:
        """Return text/image labels in requested case-ID order, never CSV order."""
        indices = [self._weak_index[case_id] for case_id in case_ids]
        artifacts = self._artifacts
        return artifacts.label_text[indices], artifacts.label_img[indices]

    @property
    def _artifacts(self) -> WeakSupervisionArtifacts:
        # Stored separately only to keep records serializable by DataLoader.
        return self.__dict__["weak_supervision"]

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        if self.image_root is None:
            raise ValueError("image_root is required when SkinCAP image paths are relative")
        return self.image_root / path


def _case_index(artifacts: WeakSupervisionArtifacts) -> dict[str, int]:
    return {case_id: index for index, case_id in enumerate(artifacts.case_ids.tolist())}


def _read_csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{path}: CSV must include a header")
            return list(reader), tuple(reader.fieldnames)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"SkinCAP CSV not found: {path}") from error


def _require_columns(fieldnames: Sequence[str], required: Sequence[str], path: Path) -> None:
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing!r}")


def _default_label_columns(fieldnames: Sequence[str]) -> tuple[str, ...]:
    return tuple(column for column in fieldnames if column.startswith("label_"))


def _disease_label(row: Mapping[str, str], columns: Sequence[str], path: Path, row_number: int) -> tuple[int, ...]:
    try:
        return tuple(int(row[column]) for column in columns)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}:{row_number}: disease label values must be integers") from error
