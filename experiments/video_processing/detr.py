import torch
import cv2 as cv
import torchvision.transforms.v2 as T
from torchvision.utils import draw_bounding_boxes
from layers.detr import model

CLASSES = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

transform = T.Compose(
    [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize(800),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


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


model.eval()

cap = cv.VideoCapture("./data/drone_cows_cut.mp4")
fourcc = cv.VideoWriter_fourcc(*"XVID") # type: ignore
out = cv.VideoWriter(f"./data/detr-out.avi", fourcc, 15.0, (1920, 1080))

orig_size = [1920, 1080]

reverse_transforms = T.Compose([T.ToDtype(torch.int, scale=True), T.Resize(orig_size)])

frame_cnt = 0

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failed on next frame")
            break

        frame_cnt += 1

        if frame_cnt % 2 == 0:
            model_in = transform(frame).unsqueeze(0)
            outs = model(model_in)
            probas = outs["pred_logits"].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.7
            if len(outs["pred_boxes"][0, keep]) > 0:
                bboxes = rescale_bboxes(outs["pred_boxes"][0, keep], orig_size)
                classes = [CLASSES[id] for id in probas[keep].argmax(1)]
                with_bb = draw_bounding_boxes(
                    T.functional.to_image(frame), bboxes, labels=classes, colors="green"
                )
                frame = with_bb.cpu().numpy().transpose((1, 2, 0))

            out.write(frame)

    cap.release()
    out.release()
    cv.destroyAllWindows()
