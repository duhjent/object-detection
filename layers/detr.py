import torch


def get_model(model_name: str) -> torch.nn.Module:
    assert model_name in {
        "detr_resnet50",
        "detr_resnet50_dc5",
        "detr_resnet101",
        "detr_resnet101_dc5",
    }
    return torch.hub.load("facebookresearch/detr:main", model_name, pretrained=True)  # type: ignore
