"""
pr1_parameters.py
Practice 1: Exploring parameters and buffers of a ResNet34 model from torchvision.

Tasks:
  (1) Count the number of parameters and buffers.
  (2) Print all parameters whose name contains "layer"
      (torchvision ResNet uses "layer1"~"layer4" as stage names).
  (3) Split parameters into two lists: 1D-or-less and 2D-or-more.
"""

import torch.nn as nn
from torchvision.models import resnet34


def load_model() -> nn.Module:
    """
    Load a ResNet34 model from torchvision without pretrained weights.

    Returns:
        model (nn.Module): ResNet34 model with random initialization.
    """
    # weights=None means no pretrained weights are downloaded
    model = resnet34(weights=None)
    return model


def count_params_and_buffers(model: nn.Module):
    """
    (1) Count the total number of parameters and buffers in the model.

    Parameters are learnable tensors (e.g., weights, biases).
    Buffers are non-learnable tensors registered via register_buffer
    (e.g., running_mean, running_var in BatchNorm).

    Args:
        model (nn.Module): The model to inspect.
    """
    pass


def print_stage_parameters(model: nn.Module, keyword: str):
    """
    (2) Print all parameters whose name contains the substring "layer".
    torchvision's ResNet34 uses "layer1" ~ "layer4" as stage names.
    Outputs the parameter name and its shape (torch.Size).

    Args:
        model (nn.Module): The model to inspect.
    """
    pass


def split_by_dimensionality(model: nn.Module, verbose: bool = True):
    """
    (3) Split all model parameters into two lists:
        - low_dim  : parameters with ndim <= 1  (scalars and 1-D tensors,
                     e.g., bias, BN weight/bias)
        - high_dim : parameters with ndim >= 2  (matrices and higher,
                     e.g., Conv2d / Linear weight tensors)

    Args:
        model (nn.Module): The model to inspect.

    Returns:
        low_dim  (list[tuple[str, Parameter]]): 1-D or lower parameters.
        high_dim (list[tuple[str, Parameter]]): 2-D or higher parameters.
    """
    return None, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load ResNet34 without pretrained weights
    model = load_model()

    # (1) Count parameters and buffers
    count_params_and_buffers(model)

    # (2) Print parameters whose name contains "layer1"
    print_stage_parameters(model, keyword="layer1")

    # (3) Split parameters by dimensionality
    low_dim_params, high_dim_params = split_by_dimensionality(model)
