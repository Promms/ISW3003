import onnx
import torch

from models.deeplabv3plus import DeepLabV3Plus


num_classes = 21
model = DeepLabV3Plus(num_classes=num_classes)
model.eval()

dummy_input = torch.randn(1, 3, 480, 640)
temp_filename = "temp_model.onnx"
final_filename = "model_structure.onnx"

torch.onnx.export(
    model,
    dummy_input,
    temp_filename,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
)

model_onnx = onnx.load(temp_filename)
for init in model_onnx.graph.initializer:
    init.ClearField("raw_data")
    init.ClearField("float_data")
    init.ClearField("int32_data")
    init.ClearField("int64_data")

onnx.save(model_onnx, final_filename)
print(f"Saved ONNX structure: {final_filename}")
