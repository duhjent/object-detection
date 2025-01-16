## Materials
https://youtu.be/wnMFBqDalnE?si=HBtn-CavmXsRSwLX
https://medium.com/@zlodeibaal/cookbook-for-edge-ai-boards-2024-2025-b9d7dcad73d6
## Raspberry Pi 5
Can achieve fast inference speed on cpu only with its arm processor. Can also connect hailo ai board, but the connection is too slow and it will just eat up most of the inference time (according to [him](https://youtu.be/wnMFBqDalnE)). But he also says that it uses pcie 1 or 2. Actually we can use pcie v3 which is twice as fast as v2. But we need to export our models into the HEF format, which could be a problem for RT-DETR. Available models are [here](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst).
## Jetson family
NVIDIA models, can use tensorRT and CUDA on them. 
## x86 based models
If they are using intel cpu or npu, that can be quite performant with openvino runtime. 
## Summary

| Chip          | Price     | Manufacturer         | Model availability                                                                 |     |
| ------------- | --------- | -------------------- | ---------------------------------------------------------------------------------- | --- |
| RPi 5         | $100      | USA/GB               | Huge with onnx or openvino                                                         |     |
| RPi 5 + Hailo | $200-250  | USA/GB               | Limited because of the HEF format. Still a lot of models available out-of-the-box. |     |
| Jetson        | $500-1000 | Waveshare is chinese | Huge.                                                                              |     |
| x86           | $50-1000  | Variety              |                                                                                    |     |
