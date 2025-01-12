from layers.retinanet import model
import torch

torch.onnx.export(
    model,
    (torch.randn((1, 3, 800, 1066)),),
    "./out/onnx/retinanet.onnx",
    input_names=["input"],
    output_names=["boxes", "labels", "scores"],
)
