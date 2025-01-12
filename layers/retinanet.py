import torch
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn,
)

model: torch.nn.Module = retinanet_resnet50_fpn(RetinaNet_ResNet50_FPN_Weights.COCO_V1)
