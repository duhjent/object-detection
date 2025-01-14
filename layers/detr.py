import torch


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def get_model(model_name: str) -> torch.nn.Module:
    assert model_name in {
        "detr_resnet50",
        "detr_resnet50_dc5",
        "detr_resnet101",
        "detr_resnet101_dc5",
    }
    return torch.hub.load("facebookresearch/detr:main", model_name, pretrained=True)  # type: ignore
