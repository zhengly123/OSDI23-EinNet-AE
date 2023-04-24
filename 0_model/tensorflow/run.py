import subprocess
import os

import sys

gpu = 0


def get_env():
    env = os.environ.copy()
    env['NVIDIA_TF32_OVERRIDE'] = '0'
    env['CUDA_VISIBLE_DEVICES'] = gpu
    return env


if not os.path.exists(f'log/{gpu}'):
    os.makedirs(f'log/{gpu}')

names = ["infogan", "fsrcnn", "csrnet", "resnet18", "gcn",  "dcgan"]
basepath = "/mnt/auxHome/models/einnet/"
subfix = ".bs{}.onnx"

onnxs = []
for bs in [1, 16]:
    for name in names:
        onnxs.append(os.path.join(basepath, name + subfix.format(bs)))
print(onnxs)

for onnx in onnxs:
    for xla in ["--xla", ""]:
        cmd = ['python3', 'run_onnx_tf.py', onnx]
        if xla:
            cmd.append(xla)
        print(" ".join(cmd))

        subprocess.run(cmd, env=get_env(),
                       stdout=open(f"log/{gpu}/{os.path.basename(onnx)}{xla}.log", "w"))

# for bs in [1, 16]:
#     for xla in ["--xla", ""]:
#         cmd = ['python3', 'longformer_tf.py', '--bs', str(bs)]
#         if xla: cmd.append(xla)
#         print(" ".join(cmd))
#         subprocess.run(cmd, env=get_env(),
#             stdout=open(f"log/{gpu}/longformer-{bs}{xla}.log", "w"))
