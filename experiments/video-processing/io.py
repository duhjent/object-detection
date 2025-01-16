import argparse

import cv2 as cv
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--input", type=str, required=True)
parser.add_argument('--nth_frame', type=int, default=2)

args = parser.parse_args()

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
            out.write(frame)

cap.release()
out.release()
cv.destroyAllWindows()
