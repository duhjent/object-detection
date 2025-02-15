import argparse
import cv2 as cv
import numpy as np
from .sort import Sort

def parse_display_size(raw: str):
    return np.array([int(x) for x in raw.split(',')])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--bboxes', type=str, required=True)
    parser.add_argument('--display-size', type=parse_display_size, default=np.array([1280, 720]))
    parser.add_argument('--nms', action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cap = cv.VideoCapture(args.input)

    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    orig_size = np.array([width, height])

    target_size = args.display_size

    scale_factor = target_size / orig_size
    scale_factor = np.array([scale_factor[0], scale_factor[1], scale_factor[0], scale_factor[1]])

    tracker = Sort(max_age=10, min_hits=3, iou_threshold=.3)

    all_classes = np.load(f'{args.bboxes}_classes.npy')
    all_boxes = np.load(f'{args.bboxes}_boxes.npy')
    all_boxes = (all_boxes * scale_factor).astype(np.int32)
    all_confs = np.load(f'{args.bboxes}_confs.npy')

    for (classes, boxes, confs) in zip(all_classes, all_boxes, all_confs):
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv.resize(frame, target_size)

        keep = confs > .3
        confs = confs[keep]
        classes = classes[keep]
        boxes = boxes[keep]

        boxes_with_confs = np.hstack((boxes, np.expand_dims(confs, -1)))

        tracklets = tracker.update(boxes_with_confs).astype(np.int32)

        for tracklet in tracklets:
            frame = cv.rectangle(frame, tracklet[:2], tracklet[2:4], (255, 0, 0), 1)
            frame = cv.putText(frame, f"id: {tracklet[-1]}", tracklet[:2], cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        
        frame = cv.putText(frame, str(len(boxes)), [100, 100], cv.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 10)

        cv.imshow("frame", frame)
        if cv.waitKey(33) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
