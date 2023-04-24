import subprocess
import os
import platform


def get_env():
    env = os.environ.copy()
    env['NVIDIA_TF32_OVERRIDE'] = '0'
    return env


if not os.path.exists(f'log'):
    os.makedirs(f'log')

names = ["infogan", "dcgan", "fsrcnn", "gcn", "resnet18", "csrnet"]
env = os.environ.copy()
basepath = env['ModelDir']  # set env var before running

print(f"model dir {basepath}")
subfix = ".bs{}.onnx"
onnxes = []
for bs in [1, 16]:
    for name in names:
        onnxes.append(os.path.join(basepath, name + subfix.format(bs)))
print(onnxes)

for onnx in onnxes:
    for xla in ["--xla", ""]:
        cmd = ['python3', 'run_onnx_tf.py', onnx]
        if xla:
            cmd.append(xla)
        print(" ".join(cmd))

        subprocess.run(cmd, env=get_env(),
                       stdout=open(f"log/tf_{platform.node()}_{os.path.basename(onnx)}{xla}.log", "w"))

# longformer
for bs in [1, 16]:
    for xla in ["--xla", ""]:
        cmd = ['python3', 'longformer_tf.py', "--bs", str(bs), ]
        if xla:
            cmd.append(xla)
        print(" ".join(cmd))

        subprocess.run(cmd, env=get_env(),
                        stdout=open(f"log/tf_{platform.node()}_longformer.bs{bs}.{xla}.log", "w"))
