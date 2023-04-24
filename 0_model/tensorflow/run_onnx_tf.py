import argparse, os, time

parser = argparse.ArgumentParser()
parser.add_argument('--xla', action='store_true')
parser.add_argument('onnx', nargs=1)

args = parser.parse_args()

if args.xla:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2'

from onnx_tf.backend import prepare

import onnx
import tensorflow as tf

onnx_model = onnx.load(args.onnx[0])

inputs = []
for input in onnx_model.graph.input:
    print(input.name)
    shape = []
    for dim in input.type.tensor_type.shape.dim:
        shape.append(dim.dim_value)

    inputs.append(tf.random.uniform(shape=shape))

tf_rep = prepare(onnx_model, device='CUDA')
model = tf_rep.run

if args.xla:
    tf.config.optimizer.set_jit(True)

warmup_num = 10
test_num = 1000

for i in range(warmup_num):
    y = model(*inputs)

t0 = time.time()
for i in range(test_num):
    y = model(*inputs)
t1 = time.time()

print(f"Impl1 Inference Time = {(t1 - t0) / test_num * 1000} ms")