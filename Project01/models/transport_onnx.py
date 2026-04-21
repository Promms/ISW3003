import torch
import onnx
from deeplabv3plus_mobilenet import DeepLabV3PLUS_MobileNet # 파일명이 다르면 수정하세요

# 1. 모델 준비
num_classes = 21 # 대회 규격에 맞는 클래스 수로 설정
model = DeepLabV3PLUS_MobileNet(num_classes=num_classes)
model.eval()

# 2. 가상 입력 생성 (조건: [1, 3, 480, 640])
dummy_input = torch.randn(1, 3, 480, 640)

# 3. ONNX로 내보내기
onnx_filename = "temp_model.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_filename, 
    export_params=True,      # 처음엔 파라미터를 포함해서 내보냅니다
    opset_version=11,        # 범용적인 버전 11 권장
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

# 4. 가중치 제거 (Diet ONNX)
model_onnx = onnx.load(onnx_filename)
for init in model_onnx.graph.initializer:
    init.ClearField("raw_data")
    init.ClearField("float_data")
    init.ClearField("int32_data")
    init.ClearField("int64_data")

# 5. 최종 구조 파일 저장
final_filename = "model_structure.onnx"
onnx.save(model_onnx, final_filename)
print(f"제출용 모델 생성 완료: {final_filename}")