"""
pr5_state.py
Practice 5: Saving, loading, and remapping state_dicts for ResNet34 from torchvision.

Tasks:
  (1) Save the model's state_dict to "model.pt".
  (2) Load the state_dict back with load_state_dict and verify the outputs match.
  (3) Wrap the model in a new nn.Module (self.module = net).
  (4) Remap the saved state_dict keys so they match the wrapped model's
      parameter names, then load it with load_state_dict.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet34


# ---------------------------------------------------------------------------
# Wrapper model
# ---------------------------------------------------------------------------

class WrappedModel(nn.Module):
    """
    A thin wrapper that stores the original network as self.module.

    When a model is wrapped this way (e.g., by nn.DataParallel or custom
    wrappers), its state_dict keys are prefixed with "module." instead of
    referring directly to the sub-layers. For example:
        Original key  : "layer1.0.conv1.weight"
        Wrapped key   : "module.layer1.0.conv1.weight"
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.module = net  # store the original model under the name "module"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_model() -> nn.Module:
    """
    Load a ResNet34 model from torchvision without pretrained weights.

    Returns:
        model (nn.Module): ResNet34 model with random initialization.
    """
    model = resnet34(weights=None)
    return model


def save_state_dict(model: nn.Module, path: str = "model.pt"):
    """
    (1) Save the model's state_dict to a file.

    state_dict() returns an OrderedDict mapping parameter/buffer names
    to their tensor values. Only parameters and buffers are saved,
    NOT the model architecture.

    Args:
        model (nn.Module): The model whose state_dict will be saved.
        path  (str)      : File path for saving (default: "model.pt").
    """
    torch.save(model.state_dict(), path)
    print("=" * 60)
    print("[Task 1] Saved state_dict")
    print("=" * 60)
    print(f"  Saved to '{path}'")
    print()


def load_and_verify(original_model: nn.Module, path: str = "model.pt"):
    """
    (2) Load the saved state_dict into a fresh model and verify that
    the outputs for a fixed input are identical to the original model.

    Args:
        original_model (nn.Module): The original model to compare against.
        path           (str)      : File path of the saved state_dict.
    """
    pass


def wrap_model(net: nn.Module) -> "WrappedModel":
    """
    (3) Wrap the given model in a WrappedModel so its parameters live
    under the "module.*" namespace.

    Args:
        net (nn.Module): The model to wrap.

    Returns:
        wrapped (WrappedModel): The wrapped model.
    """
    wrapped = WrappedModel(net)

    print("=" * 60)
    print("[Task 3] Wrapped model structure")
    print("=" * 60)
    print()
    return wrapped


def remap_and_load(wrapped_model: "WrappedModel", path: str = "model.pt"):
    """
    (4) Remap the keys in the saved state_dict by prepending "module."
    to each key, then load it into the wrapped model.

    The original state_dict has keys like "layer1.0.conv1.weight".
    The wrapped model expects keys like "module.layer1.0.conv1.weight".

    Args:
        wrapped_model (WrappedModel): The target model to load weights into.
        path          (str)         : File path of the original state_dict.
    """
    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAVE_PATH = "model.pt"

    # Load a ResNet34 model
    model = load_model()

    # (1) Save state_dict
    save_state_dict(model, path=SAVE_PATH)

    # (2) Load state_dict and verify outputs match
    loaded_model = load_and_verify(model, path=SAVE_PATH)

    # (3) Wrap the loaded model under self.module
    wrapped = wrap_model(loaded_model)

    # (4) Remap keys and load into the wrapped model
    remap_and_load(wrapped, path=SAVE_PATH)