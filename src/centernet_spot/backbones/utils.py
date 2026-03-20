from __future__ import annotations


def resolve_torchvision_weights(pretrained: bool, weights: str | None, enum_cls, default_weight):
    if not pretrained:
        return None
    if weights is None or str(weights).lower() == "default":
        return default_weight
    try:
        return enum_cls[str(weights)]
    except KeyError as exc:
        available = ", ".join(weight.name for weight in enum_cls)
        raise ValueError(f"Unknown weights '{weights}'. Available: {available}") from exc
