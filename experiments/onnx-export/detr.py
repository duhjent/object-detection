from layers.detr import get_model
import torch

for model_name in [
    "detr_resnet50",
    "detr_resnet50_dc5",
    "detr_resnet101",
    "detr_resnet101_dc5",
]:
    torch.onnx.export(
        get_model(model_name),
        (torch.randn((1, 3, 800, 1066)),),
        f"./out/onnx/{model_name}.onnx",
        input_names=["input"],
        output_names=["pred_logits", "pred_boxes"],
    )
