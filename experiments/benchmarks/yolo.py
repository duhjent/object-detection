from timeit import Timer
import numpy as np
import onnxruntime as ort
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)

args = parser.parse_args()

model_in = np.random.randn(1, 3, 640, 640).astype(np.float32)

ort_session = ort.InferenceSession(args.model)

timer = Timer(
    "ort_session.run(None, {'images': model_in})",
    f"""
import numpy as np
import onnxruntime as ort
model_in = np.random.randn(1, 3, 640, 640).astype(np.float32)
ort_session = ort.InferenceSession('{args.model}')
""",
)

print(timer.timeit(100))
