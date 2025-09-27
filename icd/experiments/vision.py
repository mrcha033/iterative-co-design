"""TorchVision-based experiment loaders."""

from __future__ import annotations

from typing import Any, Tuple

from icd.utils.imports import load_object

from ._torch_utils import resolve_device, resolve_dtype


def _resolve_weights(weights: Any, models_module: Any) -> Any:
    """Resolve a torchvision ResNet weight specifier."""

    if weights in (None, False):
        return None
    if weights is True:
        enum = getattr(models_module, "ResNet50_Weights", None)
        if enum is None:
            raise ValueError("torchvision>=0.13 is required to use pretrained weights")
        return getattr(enum, "DEFAULT")
    if isinstance(weights, str):
        enum = getattr(models_module, "ResNet50_Weights", None)
        if enum is None:
            raise ValueError("torchvision>=0.13 is required to use pretrained weights")
        try:
            return getattr(enum, weights)
        except AttributeError as exc:
            raise ValueError(f"unknown ResNet-50 weight preset '{weights}'") from exc
    return weights


def load_torchvision_resnet50(
    *,
    weights: Any = None,
    batch_size: int = 1,
    image_size: int = 224,
    device: str | None = None,
    dtype: str | None = None,
    model_loader: str | None = None,
    model_loader_kwargs: dict[str, Any] | None = None,
) -> Tuple[Any, Tuple[Any, ...]]:
    """Load a ResNet-50 and example image batch for graph construction."""

    try:
        import torch
        from torchvision import models
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch and torchvision are required for the vision loaders") from exc

    torch_dtype = resolve_dtype(dtype, torch)
    device_name = resolve_device(device, torch)

    weights_obj = _resolve_weights(weights, models)
    loader_fn = load_object(model_loader) if model_loader else models.resnet50

    loader_args = {}
    if model_loader_kwargs:
        loader_args.update(model_loader_kwargs)
    if "weights" not in loader_args:
        loader_args["weights"] = weights_obj

    model = loader_fn(**loader_args)
    model.to(device=device_name, dtype=torch_dtype)
    model.eval()

    example = torch.zeros(
        (int(batch_size), 3, int(image_size), int(image_size)),
        dtype=torch_dtype,
        device=device_name,
    )

    return model, (example,)


__all__ = ["load_torchvision_resnet50"]
