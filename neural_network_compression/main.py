"""Model compression lab — run baseline / factorization / unstructured / structured.

Usage:
    uv run main.py baseline
    uv run main.py factorize    --rank-ratio 0.5
    uv run main.py unstructured --sparsity   0.7
    uv run main.py structured   --ratio      0.3
    uv run main.py all
"""

import argparse

import torch

from compress import (
    factorize_conv2,
    structured_prune_conv2,
    unstructured_prune_conv2,
)
from data import get_test_loader
from eval import evaluate
from model import load_model
from utils import compute_flops, count_params, model_size_mb, pick_device


def _measure(model: torch.nn.Module, device: torch.device, label: str) -> None:
    loader = get_test_loader()
    acc = evaluate(model, loader, device)
    params = count_params(model)
    size = model_size_mb(model)
    flops = compute_flops(model.cpu())  # fvcore traces on CPU to avoid mps gaps
    model.to(device)
    print(
        f"[{label:>26s}] "
        f"top1={acc*100:6.2f}%  "
        f"params={params/1e6:6.2f}M  "
        f"size={size:6.2f}MB  "
        f"MACs={flops/1e6:7.1f}M"
    )


def run_baseline(device: torch.device) -> None:
    model = load_model().to(device)
    _measure(model, device, "baseline")


def run_factorize(device: torch.device, rank_ratio: float) -> None:
    model = load_model()
    factorize_conv2(model, rank_ratio)
    model.to(device)
    _measure(model, device, f"factorize r={rank_ratio}")


def run_unstructured(device: torch.device, sparsity: float) -> None:
    model = load_model()
    unstructured_prune_conv2(model, sparsity)
    model.to(device)
    _measure(model, device, f"unstructured s={sparsity}")


def run_structured(device: torch.device, ratio: float) -> None:
    model = load_model()
    structured_prune_conv2(model, ratio)
    model.to(device)
    _measure(model, device, f"structured p={ratio}")


def run_all(device: torch.device) -> None:
    print(f"# device: {device}\n")
    run_baseline(device)

    sweep = (0.1, 0.3, 0.5, 0.7, 0.9)

    print("\n## factorization (rank ratio kept)")
    for r in sweep:
        run_factorize(device, r)

    print("\n## unstructured pruning (sparsity)")
    for s in sweep:
        run_unstructured(device, s)

    print("\n## structured pruning (channel prune ratio)")
    for p in sweep:
        run_structured(device, p)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baseline")
    p_f = sub.add_parser("factorize")
    p_f.add_argument("--rank-ratio", type=float, default=0.5)
    p_u = sub.add_parser("unstructured")
    p_u.add_argument("--sparsity", type=float, default=0.5)
    p_s = sub.add_parser("structured")
    p_s.add_argument("--ratio", type=float, default=0.3)
    sub.add_parser("all")

    args = parser.parse_args()
    device = pick_device()
    if args.cmd == "baseline":
        run_baseline(device)
    elif args.cmd == "factorize":
        run_factorize(device, args.rank_ratio)
    elif args.cmd == "unstructured":
        run_unstructured(device, args.sparsity)
    elif args.cmd == "structured":
        run_structured(device, args.ratio)
    elif args.cmd == "all":
        run_all(device)


if __name__ == "__main__":
    main()
