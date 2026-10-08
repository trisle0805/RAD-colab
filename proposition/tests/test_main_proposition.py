import pytest
import torch

from main_proposition import _checkpoint_state, _validate_resume_freeze_config


def test_resume_rejects_changed_frozen_query_setting() -> None:
    with pytest.raises(ValueError, match="freeze_proposition_queries differs"):
        _validate_resume_freeze_config({"freeze_proposition_queries": True}, False)
    with pytest.raises(ValueError, match="has no saved tensor"):
        _validate_resume_freeze_config({"freeze_proposition_queries": True}, True)


def test_default_checkpoint_preserves_prior_schema_except_explicit_flag() -> None:
    module = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    generator = torch.Generator().manual_seed(4)
    checkpoint = _checkpoint_state(
        module, module, module, optimizer, scheduler, 0, {}, None, 0.0, generator, {},
        freeze_proposition_queries=False,
        frozen_proposition_queries=None,
    )

    assert checkpoint["freeze_proposition_queries"] is False
    assert "frozen_proposition_queries" not in checkpoint
    _validate_resume_freeze_config(checkpoint, False)