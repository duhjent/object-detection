from ultralytics import YOLO

yolov8_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
yolov11_models = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt']

for model_name in yolov8_models + yolov11_models:
    model = YOLO(model_name)
    model.export(format='onnx')