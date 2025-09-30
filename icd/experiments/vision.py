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


def load_efficientnet(
    *,
    variant: str = "efficientnet_b0",
    weights: Any = None,
    batch_size: int = 1,
    device: str | None = None,
    dtype: str | None = None,
    model_loader: str | None = None,
    model_loader_kwargs: dict[str, Any] | None = None,
) -> Tuple[Any, Tuple[Any, ...]]:
    """Load an EfficientNet model and example image batch.

    Args:
        variant: EfficientNet variant (e.g., "efficientnet_b0", "b0").
        weights: Weight specification (None, True, or specific preset).
        batch_size: Number of images in example batch.
        device: Target device (cuda/cpu).
        dtype: Data type (float32, float16, etc.).
        model_loader: Optional custom model loader function.
        model_loader_kwargs: Additional kwargs for model loader.

    Returns:
        Tuple of (model, (example_inputs,))
    """
    try:
        import torch
        from torchvision import models
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch and torchvision are required for vision loaders") from exc

    torch_dtype = resolve_dtype(dtype, torch)
    device_name = resolve_device(device, torch)

    # Normalize variant name
    variant_lower = variant.lower()
    if not variant_lower.startswith("efficientnet_"):
        variant_lower = f"efficientnet_{variant_lower}"

    # Get model function
    if not hasattr(models, variant_lower):
        raise ValueError(f"Unknown EfficientNet variant: {variant}")

    # Resolve weights
    if weights is None or weights is False:
        weights_obj = None
    elif weights is True:
        # Get DEFAULT weights for this variant
        weights_enum_name = f"{variant_lower.upper()}_Weights"
        weights_enum = getattr(models, weights_enum_name, None)
        if weights_enum is None:
            raise ValueError(f"No weights enum found for {variant_lower}")
        weights_obj = weights_enum.DEFAULT
    else:
        weights_obj = weights

    # Load model
    loader_fn = load_object(model_loader) if model_loader else getattr(models, variant_lower)

    loader_args = {}
    if model_loader_kwargs:
        loader_args.update(model_loader_kwargs)
    if "weights" not in loader_args:
        loader_args["weights"] = weights_obj

    model = loader_fn(**loader_args)
    model.to(device=device_name, dtype=torch_dtype)
    model.eval()

    # Variant-specific input sizes
    variant_sizes = {
        "efficientnet_b0": 224,
        "efficientnet_b1": 240,
        "efficientnet_b2": 260,
        "efficientnet_b3": 300,
        "efficientnet_b4": 380,
        "efficientnet_b5": 456,
        "efficientnet_b6": 528,
        "efficientnet_b7": 600,
    }
    image_size = variant_sizes.get(variant_lower, 224)

    example = torch.zeros(
        (int(batch_size), 3, int(image_size), int(image_size)),
        dtype=torch_dtype,
        device=device_name,
    )

    return model, (example,)


__all__ = ["load_torchvision_resnet50", "load_efficientnet"]
