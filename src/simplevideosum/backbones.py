"""Backbones: pre-trained networks that give the last convolutional activations of a frame.

A backbone is named on the command line (``--backbone``) by one of:

* an alias: ``inceptionv3`` (the default) or ``dinov2_vits14`` (DINOv2 ViT-S/14, through
  ``torch.hub``);
* the weights of any torchvision classification model, such as
  ``MobileNet_V3_Large_Weights.IMAGENET1K_V2`` (the table of weights of the torchvision
  documentation; ``.DEFAULT`` is the best weights of a model);
* the name of a torchvision classification model, such as ``mobilenet_v3_large``, for its
  default weights.

For torchvision models, the maps are the output of the last module that gives (N, C, H, W)
activations with more than one position (found by running the model once); the layers after it
(pooling, classifier) never run. The frames are resized to the crop size of the weights and
normalised with their mean and standard deviation. The vision transformers (``vit_*``) and
SqueezeNet do not follow this rule and are not supported; DenseNet maps are those of its last
block, before the ReLU that its classifier applies.
"""

import difflib
from functools import cache

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DINOV2 = "dinov2_vits14"
DINOV2_INPUT_SIZE = 224  # a multiple of the patch size 14: a 16 x 16 grid of patches
DINOV2_PATCH_SIZE = 14

# Short names of weights.
ALIASES = {"inceptionv3": "Inception_V3_Weights.IMAGENET1K_V1"}
ALIAS_NAMES = (*ALIASES, DINOV2)
EXAMPLE = "MobileNet_V3_Large_Weights.IMAGENET1K_V2"
UNSUPPORTED_MODELS = ("vit_", "squeezenet")  # prefixes of the names of the models


@cache
def _torchvision_weights() -> dict[str, tuple[str, object]]:
    """Every pre-trained classification weights of torchvision, by ``"Class_Weights.NAME"``, with
    the name of their model. Importing torchvision is slow, hence the deferred import."""
    from torchvision import models

    table = {}
    for model_name in models.list_models(module=models):
        weights_enum = models.get_model_weights(model_name)
        if model_name.startswith(UNSUPPORTED_MODELS):
            continue
        for member_name, weights in weights_enum.__members__.items():
            if member_name != "DEFAULT":  # an alias of the best weights
                table[f"{weights_enum.__name__}.{member_name}"] = (model_name, weights)
    return table


def torchvision_weights(spec: str) -> tuple[str, object]:
    """The model name and the weights that ``spec`` designates (raises ValueError)."""
    from torchvision import models

    spec = ALIASES.get(spec, spec)
    if spec.lower().startswith(UNSUPPORTED_MODELS):
        raise ValueError(
            f"'{spec}' is not supported: the vision transformers and SqueezeNet do not give the "
            "last convolutional activations like the other models"
        )
    table = _torchvision_weights()
    if spec in table:
        return table[spec]
    names = [n for n in models.list_models(module=models) if not n.startswith(UNSUPPORTED_MODELS)]
    if spec in names:  # a model: its default weights
        return spec, models.get_model_weights(spec).DEFAULT
    try:
        weights = models.get_weight(spec)  # "Class_Weights.DEFAULT"
    except (ValueError, AttributeError, KeyError):
        weights = None
    if weights is not None:
        model_name = next(name for name, w in table.values() if type(w) is type(weights))
        return model_name, weights
    close = difflib.get_close_matches(spec, [*table, *names], n=3)
    hint = f" Did you mean {', '.join(close)}?" if close else ""
    raise ValueError(
        f"unknown backbone '{spec}' (available: {', '.join(ALIAS_NAMES)}, or the weights of a "
        f"torchvision classification model such as {EXAMPLE}).{hint}"
    )


def validate_backbone(spec: str) -> None:
    """Raise ValueError unless ``spec`` designates a backbone."""
    if spec not in ALIAS_NAMES:
        torchvision_weights(spec)


class _Captured(Exception):
    """Carries the output of the module where a forward pass is stopped."""

    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output


def _stop(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
    raise _Captured(output)


def find_maps_module(model: nn.Module, input_size: int) -> str:
    """Name of the module whose output is the last convolutional activations of ``model``: the
    last one to give (N, C, H, W) activations with more than one position, in a forward pass."""
    outputs, hooks = [], []
    for name, module in model.named_modules():
        if name:
            hooks.append(
                module.register_forward_hook(lambda m, i, o, name=name: outputs.append((name, o)))
            )
    try:
        with torch.inference_mode():
            model(torch.zeros(1, 3, input_size, input_size))
    finally:
        for hook in hooks:
            hook.remove()
    names = [
        name
        for name, o in outputs
        if isinstance(o, torch.Tensor) and o.ndim == 4 and o.shape[2] > 1 and o.shape[3] > 1
    ]
    if not names:
        raise ValueError(f"no convolutional activations found in {type(model).__name__}")
    return names[-1]


class TorchvisionMaps(nn.Module):
    """A torchvision classification model up to its last convolutional activations:
    (N, 3, S, S) frames to (N, C, H, W) maps, with S the ``input_size``. The forward pass is
    stopped at that module (see ``find_maps_module``), so the layers after it do not run."""

    def __init__(
        self,
        model: nn.Module,
        maps_module: str,
        input_size: int,
        mean: tuple[float, ...],
        std: tuple[float, ...],
    ) -> None:
        super().__init__()
        self.model = model
        self.maps_module = maps_module
        self.input_size, self.mean, self.std = input_size, tuple(mean), tuple(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        handle = self.model.get_submodule(self.maps_module).register_forward_hook(_stop)
        try:
            self.model(x)
        except _Captured as captured:
            return captured.output
        finally:
            handle.remove()
        raise RuntimeError(
            f"the forward pass did not reach '{self.maps_module}'"
        )  # pragma: no cover


def build_torchvision_backbone(spec: str, pretrained: bool = True) -> nn.Module:
    """The maps of a torchvision classification model, with its pre-trained weights if
    ``pretrained``, downloaded on first use (random weights otherwise, which is only useful to
    test). The input size and normalisation are those of the weights."""
    from torchvision import models

    model_name, weights = torchvision_weights(spec)
    preset = weights.transforms()
    input_size = max(preset.crop_size)
    kwargs = {"progress": True} if pretrained else {}
    if not pretrained and model_name == "inception_v3":
        kwargs["init_weights"] = False  # the default initialisation is slow
    model = models.get_model(model_name, weights=weights if pretrained else None, **kwargs)
    model.eval()
    return TorchvisionMaps(
        model, find_maps_module(model, input_size), input_size, preset.mean, preset.std
    )


class DinoV2Maps(nn.Module):
    """DINOv2 up to its last block: (N, 3, 224, 224) frames to the (N, C, 16, 16) map of the
    normalised patch tokens (the class token is dropped)."""

    input_size = DINOV2_INPUT_SIZE
    mean, std = IMAGENET_MEAN, IMAGENET_STD

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(x)["x_norm_patchtokens"]  # (N, 16 * 16, C)
        side = x.shape[-1] // DINOV2_PATCH_SIZE
        return tokens.transpose(1, 2).reshape(len(x), -1, side, side)


def build_dinov2_vits14(pretrained: bool = True) -> nn.Module:
    """DINOv2 ViT-S/14 (22M parameters, 384 channels), from the authors' repository through
    ``torch.hub``: the first use downloads its code from GitHub (``facebookresearch/dinov2``,
    main branch) and, if ``pretrained``, its weights, both to the torch hub cache."""
    model = torch.hub.load(
        "facebookresearch/dinov2", DINOV2, pretrained=pretrained, trust_repo=True
    )
    return DinoV2Maps(model)


def build_backbone(spec: str, pretrained: bool = True) -> nn.Module:
    """The backbone designated by ``spec`` (see the module docstring); its ``input_size``,
    ``mean`` and ``std`` attributes say how to prepare the frames."""
    if spec == DINOV2:
        return build_dinov2_vits14(pretrained)
    return build_torchvision_backbone(spec, pretrained)
