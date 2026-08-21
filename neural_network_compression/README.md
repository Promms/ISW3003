# Neural Network Compression

CIFAR-100 ResNet-50을 대상으로 모델 압축 방법을 실습한 과제 코드입니다.

## Status

| Method | Status |
| --- | --- |
| Baseline evaluation | available |
| Low-rank factorization | implemented |
| Unstructured pruning | not implemented in the archived version |
| Structured channel pruning | not implemented in the archived version |

현재 저장된 코드에서 실행 가능한 압축 방법은 low-rank factorization입니다. 두 pruning 모듈에는 미구현 부분이 남아 있습니다.

## Run

```bash
uv sync
uv run main.py baseline
uv run main.py factorize --rank-ratio 0.5
```

실행 시 top-1 accuracy, parameter 수, 모델 크기와 MACs를 출력합니다. 실험 출력 파일은 저장되어 있지 않아 결과표는 포함하지 않았습니다.
