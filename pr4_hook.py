"""
pr4_hook.py
Practice 4: Registering and removing forward hooks on ResNet34 from torchvision.

Tasks:
  (1) Define a forward_pre_hook that prints (module name, input shape)
      for every nn.Conv2d layer.
  (2) Register the hook on all Conv2d layers and run a forward pass to verify.
  (3) Remove all registered hooks.
  (4) Define a forward_hook that prints (module name, input shape, output shape)
      for every nn.Conv2d layer, then register and verify it.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet34


def load_model() -> nn.Module:
    """
    Load a ResNet34 model from torchvision without pretrained weights.

    Returns:
        model (nn.Module): ResNet34 model with random initialization.
    """
    model = resnet34(weights=None)
    return model


def make_pre_hook(module_name: str):
    """
    Factory function that creates a forward_pre_hook closure.

    A forward_pre_hook is called BEFORE the module's forward() method.
    Signature: hook(module, input) -> None or modified input

    Args:
        module_name (str): The name of the Conv2d module (for display).

    Returns:
        hook (callable): The pre-hook function.
    """
    def hook(module: nn.Module, input: tuple):
        pass
    return hook


def make_post_hook(module_name: str):
    """
    Factory function that creates a forward_hook closure.

    A forward_hook is called AFTER the module's forward() method.
    Signature: hook(module, input, output) -> None or modified output

    Args:
        module_name (str): The name of the Conv2d module (for display).

    Returns:
        hook (callable): The post-hook function.
    """
    def hook(module: nn.Module, input: tuple, output: torch.Tensor):
        pass
    return hook


def register_pre_hooks(model: nn.Module) -> list:
    """
    (1)+(2) Register a forward_pre_hook on every nn.Conv2d submodule.

    register_forward_pre_hook() returns a RemovableHook handle.
    Storing these handles allows us to remove hooks later.

    Args:
        model (nn.Module): The model to attach hooks to.

    Returns:
        handles (list): List of RemovableHook handles.
    """
    pass


def remove_hooks(handles: list):
    """
    (3) Remove all previously registered hooks using their handles.

    Calling handle.remove() detaches the hook from the module.

    Args:
        handles (list): List of RemovableHook handles to remove.
    """
    pass  # use handle.remove()


def register_post_hooks(model: nn.Module) -> list:
    """
    (4) Register a forward_hook on every nn.Conv2d submodule.

    Args:
        model (nn.Module): The model to attach hooks to.

    Returns:
        handles (list): List of RemovableHook handles.
    """
    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create a dummy input: batch=1, 3-channel, 224x224 image
    dummy_input = torch.randn(1, 3, 224, 224)

    # -----------------------------------------------------------------------
    # (1) + (2) Register forward_pre_hooks and run a forward pass
    # -----------------------------------------------------------------------
    model = load_model()

    print("=" * 60)
    print("[Task 1+2] forward_pre_hook — (module name, input shape)")
    print("=" * 60)
    pre_handles = register_pre_hooks(model)

    with torch.no_grad():
        _ = model(dummy_input)
    print()

    # -----------------------------------------------------------------------
    # (3) Remove all pre-hooks
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("[Task 3] Removing all pre-hooks")
    print("=" * 60)
    remove_hooks(pre_handles)

    # Verify hooks are gone: a second forward pass should produce no output
    with torch.no_grad():
        _ = model(dummy_input)
        
    # -----------------------------------------------------------------------
    # (4) Register forward_hooks and run a forward pass
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("[Task 4] forward_hook — (module name, input shape, output shape)")
    print("=" * 60)
    post_handles = register_post_hooks(model)

    with torch.no_grad():
        _ = model(dummy_input)

    # Clean up post-hooks as well
    remove_hooks(post_handles)