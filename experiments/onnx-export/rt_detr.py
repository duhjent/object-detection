from transformers import RTDetrForObjectDetection 
import torch

# INFO: used this: https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/tools/export_onnx.py
# Probably it is better for export, maybe some nuances are handled I didn't think of
model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")

torch.onnx.export(
    model.model,
    (torch.randn((1, 3, 640, 640)),),
    "./out/onnx/rtdetr_resnet50.onnx",
    input_names=["input"],
    output_names=["enc_topk_logits", "enc_topk_bboxes"],
)
