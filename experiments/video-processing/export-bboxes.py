import cv2 as cv
import numpy as np
import onnxruntime as ort
import argparse
from tqdm import tqdm

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--provider", type=str, default="CPUExecutionProvider")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    ort_session = ort.InferenceSession(args.model)

    cap = cv.VideoCapture(args.input)

    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    orig_size = [width, height]
    total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)

    # f = open(args.out, 'w')
    all_classes, all_boxes, all_confs = [], [], []

    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            pbar.update(1)

            model_in = cv.resize(frame, (640, 640))
            model_in = model_in / 255.0
            model_in = model_in.transpose((2, 0, 1))
            model_in = np.expand_dims(model_in, 0).astype(np.float32)

            classes, boxes, confs = ort_session.run(
                None, {"images": model_in, "orig_target_sizes": np.array([orig_size])}
            )
            all_classes.append(classes)
            all_boxes.append(boxes)
            all_confs.append(confs)

        np.save(f'{args.out}_classes.npy', np.vstack(all_classes))
        np.save(f'{args.out}_boxes.npy', np.vstack(all_boxes))
        np.save(f'{args.out}_confs.npy', np.vstack(all_confs))

    cap.release()

if __name__ == '__main__':
    main()

