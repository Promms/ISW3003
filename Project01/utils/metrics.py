import torch
from torch import Tensor

IGNORE_INDEX = 255


def accuracy(logits: Tensor, targets: Tensor) -> float:
    """평가용: 즉시 Python float 반환 (GPU sync 발생). val 루프에서만 사용."""
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        mask = (targets != IGNORE_INDEX)
        correct = (pred[mask] == targets[mask])
        return correct.float().mean().item() * 100.0


@torch.no_grad()
def accuracy_counts(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    """
    학습 루프용: (correct, total) 을 GPU 스칼라 텐서로 반환.
    .item() 호출이 없어 CUDA 파이프라인을 막지 않음.
    호출 측에서 누적한 뒤 log_interval 마다만 .item() 하면 sync 1/N 로 감소.
    """
    pred = logits.argmax(dim=1)
    mask = (targets != IGNORE_INDEX)
    correct = (pred[mask] == targets[mask]).sum()
    total   = mask.sum()
    return correct, total

def mIoU(logits: Tensor, targets: Tensor, num_class: int) -> float:

    with torch.no_grad():
        pred = logits.argmax(dim=1) 
        mask = (targets != IGNORE_INDEX)  

        pred_flat = pred[mask]                          # 유효한 픽셀만 masking해서 1D로 펼침
        targ_flat = targets[mask]                       # 동일

        index = targ_flat * num_class + pred_flat     # 각 class에 대해서 어떻게 예측했는지
        conf = torch.bincount(index, minlength=num_class*num_class).reshape(num_class, num_class)

        True_Positive = conf.diag()
        False_Negative = conf.sum(dim=1) - True_Positive    # (행 방향 Sum: 정답이 c인 픽셀 합) - 정답 횟수 -> 정답인데 예측 못함
        False_Positive = conf.sum(dim=0) - True_Positive    # (열 방향 Sum: c로 예측한 픽셀 합) - 정답 횟수 -> 예측 했는데 틀린

        IoU = True_Positive / (True_Positive + False_Positive + False_Negative)

        valid = (conf.sum(dim=1) + conf.sum(dim=0)) > 0     # 한 번도 등장 X(정답에도 X, 예측에도 X)인 class는 제외
        mIoU = IoU[valid].mean().item()

        return mIoU


class AverageMeter:

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0
