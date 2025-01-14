import argparse

import cv2 as cv
import numpy as np
import onnxruntime as ort
import torch
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from datasets import CLASSES

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--provider", type=str, default="CPUExecutionProvider")
parser.add_argument('--nth_frame', type=int, default=2)

args = parser.parse_args()

ort_session = ort.InferenceSession(args.model, providers=[args.provider])

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

        if frame_cnt % args.nth_frame == 1:
            continue

        model_in = cv.resize(frame, (640, 640))
        model_in = model_in / 255.0
        # model_in = (model_in - mean) / std
        model_in = model_in.transpose((2, 0, 1))
        model_in = np.expand_dims(model_in, 0).astype(np.float32)

        classes, boxes, confs = ort_session.run(
            None, {"images": model_in, "orig_target_sizes": np.array([orig_size])}
        )
        keep = confs > 0.25

        classes = classes[keep]
        boxes = boxes[keep]
        confs = confs[keep]

        if len(boxes) > 0:
            classes = [CLASSES[id] for id in classes]
            with_bb = draw_bounding_boxes(
                T.functional.to_image(frame),
                torch.tensor(boxes),
                labels=classes,
                colors="red",
                width=3,
            )
            frame = with_bb.cpu().numpy().transpose((1, 2, 0))

        out.write(frame)

cap.release()
out.release()
cv.destroyAllWindows()
