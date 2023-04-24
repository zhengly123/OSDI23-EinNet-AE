import os
import sys

import logging
import numpy as np
import tvm
from tvm import te, auto_scheduler, topi, autotvm
from tvm.topi.testing import conv2d_nchw_python
import tvm.testing
import re

######################################################################
# Define the computation
dtype = "float32"
target = "cuda"
log_filename = "op.autotvm.log"

np.random.seed(0)


def op_batch_matmul(B, M, N, K):
    A = te.placeholder((B, M, K), name="A")
    B = te.placeholder((B, N, K), name="B")
    out = topi.nn.batch_matmul(A, B)
    return ("batch_matmul.cuda", [A, B])


def op_conv2D(n, c, h, w, f, r, s, padding, stride, dilation):
    data = te.placeholder((n, c, h, w), name="data")
    kernel = te.placeholder((f, c, r, s), name="kernel")
    # bias = te.placeholder((1, CO, 1, 1), name="bias")
    # out = topi.nn.conv2d_nchw(
    #     data, kernel, stride, padding, dilation=dilation, out_dtype=dtype)
    return ("conv2d_nchw.cuda", [data, kernel, stride, padding, dilation])


def op_conv2D_winograd(n, c, h, w, f, r, s, padding, stride, dilation):
    data = te.placeholder((n, c, h, w), name="data")
    kernel = te.placeholder((f, c, r, s), name="kernel")
    # bias = te.placeholder((1, CO, 1, 1), name="bias")
    # out = topi.nn.conv2d_winograd_nhwc(
    #     data, kernel, stride, padding, dilation, out_dtype=dtype)
    # data, kernel, strides, padding, dilation, out_dtype
    return ("conv2d_nchw_winograd.cuda", [data, kernel, (stride, stride), padding, dilation, dtype])


def op_conv2D_transpose(n, c, h, w, f, r, s, padding, stride, dilation, output_padding=(0, 0)):
    assert(dilation == 1)
    data = te.placeholder((n, f, h, w), name="data")
    kernel = te.placeholder((f, c, r, s), name="kernel")
    # out = topi.nn.conv2d_transpose_nchw(data, kernel, strides=(
    #     stride, stride), padding=padding, out_dtype=dtype, output_padding=output_padding)
    return ("conv2d_transpose_nchw.cuda", [data, kernel, (stride, stride), padding, dtype, output_padding])

def print_best_time(tuner, inputs, results):
    if not hasattr(tuner, 'best_time'):
        tuner.best_time = 999999999
    for k, (inp, res) in enumerate(zip(inputs, results)):
        config = inp.config
        if res.error_no == 0:
            flops = inp.task.flop / np.mean(res.costs)
            error_ct = 0
            result_msg = res
            t = np.mean(res.costs)
            tuner.best_time = min(tuner.best_time, t)
    print(f'===== best time {tuner.best_time * 1000} ms, perf {tuner.best_flops} Gflops, FLOPs {tuner.best_time * tuner.best_flops}')

def autotvm_tune(tasks):
    with tvm.target.Target(target):
        # logging config (for printing tuning log to screen)
        logging.getLogger("autotvm").setLevel(logging.DEBUG)
        logging.getLogger("autotvm").addHandler(
            logging.StreamHandler(sys.stdout))
        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(
                repeat=10, min_repeat_ms=100, timeout=4),
        )
        for i, task in enumerate(tasks):
            print(f'===== Task {i}/{len(tasks)}:')
            print(f'Config space = {task.config_space}')
            tuner = autotvm.tuner.XGBTuner(task)
            tuner.tune(
                n_trial=1024,
                measure_option=measure_option,
                callbacks=[autotvm.callback.log_to_file(log_filename), print_best_time],
            )
            # inspect the best config
            dispatch_context = autotvm.apply_history_best(log_filename)
            best_config = dispatch_context.query(task.target, task.workload)
            print("\nBest config:")
            print(best_config)


task_parameters = [
    # Conv (n, c, h, w, f, r, s, padding, stride, dilation),
    # Conv3x3 -> Gemm # ResNet-18
    (op_conv2D, (1, 512, 7, 7, 512, 3, 3, 1, 1, 1), "Conv3x3_origin"),
    (op_batch_matmul, (1, 1*7*7, 512*3*3, 512), "Conv3x3_opt"),

    # ConvTranpose -> Gemm # InfoGAN CelebA_ConvTranspose_3
    (op_conv2D_transpose, (16, 256, 2, 2, 448, 4, 4, 1, 2, 1), "ConvTranspose"),
    (op_batch_matmul, (1, 64, 4096, 448), "ConvTranspose_opt"),

    # Conv5x5 -> Conv3x3 # SRCNN
    (op_conv2D, (16, 32, 224, 224, 1, 5, 5, 2, 1, 1), "Conv5x5_origin"),
    (op_conv2D, (16, 32, 224, 224, 4, 3, 3, 2, 1, 1), "Conv5x5_opt"),
    (op_conv2D_winograd, (16, 32, 224, 224, 4, 3, 3,
     2, 1, 1), "Conv5x5_opt_winograd"),  # Winograd
]

tasks = []
for fn, params, desc in task_parameters:
    print(params)
    task_name, args = fn(*params)
    tasks.append(autotvm.task.create(task_name, args, target))
print('# of tasks = %d' % (len(tasks)))

# Tuning
autotvm_tune(tasks)

# # Evaluating
# for task in tasks:
#     ansor_load(task)
# #     # ansor_dump_all_logs(task)
#     print('\n'*5)

# Customize the topi schedule (disable loop unroll)
# for params in task_parameters:
#     mock_ansor_schedule(*params)
