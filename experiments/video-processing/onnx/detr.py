import argparse

import cv2 as cv
import numpy as np
import onnxruntime as ort
import torch
from scipy.special import softmax
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from datasets import CLASSES


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    b = [
        x[:, 0] - 0.5 * x[:, 2],
        x[:, 1] - 0.5 * x[:, 3],
        x[:, 0] + 0.5 * x[:, 2],
        x[:, 1] + 0.5 * x[:, 3],
    ]
    return np.vstack(b).T


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * np.array([img_w, img_h, img_w, img_h])
    return b


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--input", type=str, required=True)
parser.add_argument('--nth_frame', type=int, default=2)

args = parser.parse_args()

ort_sesssion = ort.InferenceSession(args.model)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

cap = cv.VideoCapture(args.input)
fourcc = cv.VideoWriter_fourcc(*"XVID")  # type: ignore
out = cv.VideoWriter(args.out, fourcc, 15.0, (1920, 1080))

orig_size = [1920, 1080]

frame_cnt = 0
total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)


with tqdm(total=total_frames) as pbar:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_cnt += 1
        pbar.update(1)

        if frame_cnt % args.nth_frame == 0:
            model_in = cv.resize(frame, (1422, 800)) / 255.0
            model_in = (model_in - mean) / std
            model_in = np.expand_dims(model_in.transpose((2, 0, 1)), 0).astype(
                np.float32
            )
            pred_logits, pred_boxes = ort_sesssion.run(
                ["pred_logits", "pred_boxes"], {"input": model_in}
            )
            probas = softmax(pred_logits, axis=2)[0, :, :-1]
            keep = probas.max(-1) > 0.7
            if len(pred_boxes[0, keep]) > 0:
                bboxes = rescale_bboxes(pred_boxes[0, keep], orig_size)
                classes = [CLASSES[id] for id in probas[keep].argmax(1)]
                with_bb = draw_bounding_boxes(
                    T.functional.to_image(frame),
                    torch.tensor(bboxes),
                    labels=classes,
                    colors="red",
                    width=3,
                )
                frame = with_bb.cpu().numpy().transpose((1, 2, 0))

            out.write(frame)

cap.release()
out.release()
cv.destroyAllWindows()
