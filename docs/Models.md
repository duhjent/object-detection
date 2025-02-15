## Faster R-CNN
[Paper](https://arxiv.org/pdf/1506.01497) Two-stage detector, well-known architecture. Appeared in 2016, may be outdated, but  has undergone many design iterations and its performance was greatly improved since the original publication with FPNs etc. Included in detectron2. Not for real-time.
## RetinaNet
[Paper](https://arxiv.org/pdf/1708.02002v2) An SSD framework based model, but heavily extended and modified. Extended by FPNs for better detection of different scale objects. Appeared in 2017, may be outdated.
Included in detectron2. Not for real-time.
## DETR
[Paper](https://arxiv.org/pdf/2005.12872) Transformer-based model by facebook. Apache license. Can be used with resnet50 and resnet101 backbones, shows high mAP, problem is training process (already trained) and inference performance (see below). Included in torchHub. Easy to implement. Not for real-time.
## RT-DETR
[Paper](https://arxiv.org/pdf/2304.08069) Real-time detr. Authored by chinese scientists from Peking university.
Code for export onnx [here](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/tools/export_onnx.py).
Is available in openvino model zoo, in huggingface etc., has many stars and the paper has 400+ citations.
Export to openvino: https://github.com/nanmi/RT-DETR-Deploy.
Apache license.

## Summary
FPS is from DETR paper, AP are mAP for COCO validation dataset.

| Model                | FPS | AP   | AP_s | AP_m | AP_l |
| -------------------- | --- | ---- | ---- | ---- | ---- |
| Faster R-CNN-DC5     | 16  | 41.1 | 22.9 | 45.9 | 55   |
| Faster R-CNN-FPN     | 26  | 42   | 26.6 | 45.4 | 53.4 |
| Faster RCNN-R101-FPN | 20  | 44   | 27.2 | 48.1 | 56   |
| DETR                 | 28  | 42   | 20.5 | 45.8 | 61.1 |
| DETR-DC5             | 12  | 43.3 | 22.5 | 47.3 | 61.1 |
| DETR-R101            | 20  | 43.5 | 21.9 | 48   | 61.8 |
| DETR-DC5-R101        | 10  | 44.9 | 23.7 | 49.5 | 62.3 |
| RetinaNet-R101       | -   | 39.1 | 21.8 | 42.7 | 50.2 |

Another table, found in the RT-DETR paper:
![[Pasted image 20250113222228.png]]

Performance with ONNX runtime

| Model         | Short video CPU AMD (local) | Short video CPU Intel i5-14500 (vast.ai) | Long video GPU (vast.ai) |
| ------------- | --------------------------- | ---------------------------------------- | ------------------------ |
| I/O time      | 00:10                       | 00:05                                    | 00:41                    |
| YOLOv8n       | 00:54                       | 00:26                                    | 01:38                    |
| YOLOv8s       | 01:15                       | 00:34                                    | 01:48                    |
| YOLOv8m       | 02:09                       | 00:59                                    | 01:50                    |
| YOLOv8l       | 03:38                       | 01:42                                    | 01:55                    |
| YOLOv8x       | 05:29                       | 02:34                                    | 02:01                    |
| YOLO11n       | 00:47                       | 00:25                                    | 01:41                    |
| YOLO11s       | 01:16                       | 00:34                                    | 01:50                    |
| YOLO11m       | 02:16                       | 01:00                                    | 01:53                    |
| YOLO11l       | 03:14                       | 01:14                                    | 01:54                    |
| YOLO11x       | 05:20                       | 02:09                                    | 01:56                    |
| DETR-R50      | 05:39                       | 02:26                                    | 03:04                    |
| DETR-R50-DC5  | 16:03                       | 06:52                                    | 03:58                    |
| DETR-R101     | 08:15                       | 03:49                                    | 03:11                    |
| DETR-R101-DC5 | 18:56                       | 07:33                                    | 04:04                    |
| RT-DETR-R34   | 03:04                       | 01:25                                    | 02:10                    |
| RT-DETR-R50   | 04:25                       | 03:19                                    | 02:07                    |
| RT-DETR-R101  | 05:48                       | 04:24                                    | 02:21                    |
