"""Explicit, auditable BioClinicalBERT adaptation policy for Round B1."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from torch import nn

AdaptationMode = str


def bert_encoder_layers(text_encoder: nn.Module) -> list[nn.Module]:
    """Return the native HuggingFace BERT encoder layers or fail loudly."""
    bert_model = getattr(text_encoder, "bert_model", None)
    layers = getattr(getattr(bert_model, "encoder", None), "layer", None)
    if layers is None or not isinstance(layers, nn.ModuleList) or not layers:
        raise ValueError("text_encoder.bert_model.encoder.layer must be a non-empty ModuleList")
    return list(layers)


def selected_bert_layers(text_encoder: nn.Module, adaptation: AdaptationMode) -> list[nn.Module]:
    """Return the allowed BERT layers for a supported adaptation mode."""
    if adaptation == "frozen":
        return []
    if adaptation not in {"last1", "last2"}:
        raise ValueError("bert_adaptation must be 'frozen', 'last1', or 'last2'")
    layers = bert_encoder_layers(text_encoder)
    count = 1 if adaptation == "last1" else 2
    if len(layers) < count:
        raise ValueError(f"bert_adaptation={adaptation} requires at least {count} encoder layers")
    return layers[-count:]


def configure_text_encoder_adaptation(text_encoder: nn.Module, adaptation: AdaptationMode) -> list[nn.Module]:
    """Freeze the encoder, then unfreeze exactly the selected final layers."""
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    selected = selected_bert_layers(text_encoder, adaptation)
    for layer in selected:
        layer.requires_grad_(True)
    return selected


def enforce_text_encoder_modes(text_encoder: nn.Module, selected: Iterable[nn.Module], training: bool) -> None:
    """Keep frozen BERT components deterministic while selected layers may train."""
    selected_ids = {id(module) for module in selected}
    text_encoder.eval()
    for layer in bert_encoder_layers(text_encoder):
        layer.train(training and id(layer) in selected_ids)
    for layer in bert_encoder_layers(text_encoder):
        expected = training and id(layer) in selected_ids
        if layer.training != expected:
            raise RuntimeError("BERT layer train/eval policy could not be enforced")


def named_text_components(model: nn.Module) -> dict[str, nn.Module]:
    """Return independently auditable text components in stable names."""
    text_encoder = getattr(model, "text_encoder")
    bert_model = getattr(text_encoder, "bert_model", None)
    result: dict[str, nn.Module] = {
        "text_cross_attention": getattr(model, "text_cross_attention"),
        "text_head": getattr(model, "text_head"),
    }
    embeddings = getattr(bert_model, "embeddings", None)
    if isinstance(embeddings, nn.Module):
        result["bert_embeddings"] = embeddings
    for index, layer in enumerate(bert_encoder_layers(text_encoder)):
        result[f"bert_encoder_layer_{index}"] = layer
    pooler = getattr(bert_model, "pooler", None)
    if isinstance(pooler, nn.Module):
        result["bert_pooler"] = pooler
    return result


def module_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def component_hashes(model: nn.Module) -> dict[str, str]:
    return {name: module_sha256(module) for name, module in named_text_components(model).items()}