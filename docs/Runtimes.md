## PyTorch
Default runtime, the deep learning library we use for modelling networks and training them.
License is probably fully open source
## ONNX
Framework-independent format. Simple to get from PyTorch since it is built into PyTorch. And simple to run inference.
The library is apache license and the runtime is mit license.
## NCNN
ONNX-like thing by Tencent. First it was PNNX (PyTorch NNX). But the tutorials seem to be messy and overall the runtimes doesn't feel stable or well-established.
Licensed under BSD 3.
## OpenVINO
Runtime for x86 cpu gpu and npu (intel, arm, intel iris etc. gpus, inter core ultra). Can use pytorch models natively and contains rt-detr in its model zoo. Also available as ONNX execution provider.
Can use quantization and other post-training tricks for beter inference performance (see [here](https://github.com/openvinotoolkit/openvino_notebooks/blob/383cfbd020483f2b1c52ff0daf1a47b192732770/notebooks/depth-anything/depth-anything.ipynb)).
## Runtime comparison for yolov11 on RPi 5
![[rpi-yolo11-benchmarks.avif]]
RPi5 yolov11 nano and small inference time for different runtimes (can use either PyTorch on ONNX).