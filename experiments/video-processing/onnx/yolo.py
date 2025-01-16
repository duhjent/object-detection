import argparse

import cv2 as cv
import numpy as np
import onnxruntime as ort
import torch
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from datasets import CLASSES


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--provider", type=str, default="CPUExecutionProvider")
parser.add_argument("--nth_frame", type=int, default=2)

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

        if frame_cnt % args.nth_frame == 0:
            model_in = cv.resize(frame, (640, 640)) / 255.0
            model_in = np.expand_dims(model_in.transpose((2, 0, 1)), 0).astype(
                np.float32
            )
            outs = np.squeeze(ort_session.run(None, {'images': model_in})[0]).T

            boxes_cxcywh = outs[:,:4].astype(np.int32)
            boxes_xywh = np.vstack((boxes_cxcywh[:,0] - (boxes_cxcywh[:,2] / 2), boxes_cxcywh[:,1] - (boxes_cxcywh[:,3] / 2), boxes_cxcywh[:,2], boxes_cxcywh[:,3])).T
            boxes_xyxy = np.vstack((boxes_xywh[:,0], boxes_xywh[:,1], boxes_xywh[:,0] + boxes_xywh[:,2], boxes_xywh[:,1] + boxes_xywh[:,3])).T

            labels = outs[:,4:]
            conf = labels.max(1)

            nms = cv.dnn.NMSBoxes(boxes_xywh, conf, .25, .7) # type: ignore

            boxes_xyxy = boxes_xyxy[nms,:]
            labels = labels[nms,:]

            if len(boxes_xyxy) > 0:
                bboxes = boxes_xyxy / np.array([640, 640, 640, 640]) * np.array([orig_size[0], orig_size[1], orig_size[0], orig_size[1]])
                classes = [CLASSES[id] for id in labels.argmax(1)]
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
