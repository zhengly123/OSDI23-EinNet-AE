import timeit
import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing
import math

import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import topi, autotvm
import logging
from datetime import datetime
import sys
# Enable debug logs
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <cpu/gpu> <log>")
    exit(-1)
if sys.argv[1] == 'cpu':
    target_name = 'llvm -libs=mkl -mcpu=core-avx2'
elif sys.argv[1] == 'gpu':
    target_name = 'cuda -libs=cublas'
else:
    assert(False)

tuning_rounds = 1000

n_heads = 8
seq_len = 5000
feat_len = 512
w = 1000
dilation = 2  # counts from 1

target = tvm.target.Target(target_name)
dtype, itype = 'float32', 'int32'
time_now = datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
log_file = sys.argv[2]
tune = False
print('Mode = ', 'Tuning' if tune else 'Evaluation')
print('log file:', log_file)


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def longformer_compute0(bs, n_heads, seq_len, feat_len, w, dilation, dilation_heads, dtype):
    assert(feat_len % n_heads == 0)
    head_size = feat_len // n_heads
    q = te.placeholder((bs*n_heads, seq_len, head_size),
                       name="q", dtype=dtype)
    k = te.placeholder((bs*n_heads, seq_len, head_size),
                       name="k", dtype=dtype)
    pad = dilation*w
    k_pad = te.compute(
        (bs*n_heads, seq_len+2*pad, head_size),
        lambda a, b, c: tvm.tir.if_then_else(
            tvm.tir.all(b >= pad, b - pad < seq_len),
            k[a, b - pad, c],
            tvm.tir.const(0.0, dtype),
        ),
        name="Kpad",
    )

    p = te.reduce_axis((0, head_size), name="p")
    prob = te.compute(
        (bs*n_heads, seq_len, 2*w+1),
        # TVM unsupport: Reductions are only allowed at the top level of compute
        # lambda i, j, k: tvm.tir.if_then_else(i < dilation_heads,
        #                                      te.sum(
        #                                          q[i, j, p]*k_pad[i, j+dilation*(k-w), p], axis=p),
        #                                      te.sum(q[i, j, p]*k_pad[i, j+(k-w), p], axis=p)),
        lambda i, j, k:
        te.sum(tvm.tir.if_then_else(i < dilation_heads,
                                    q[i, j, p]*k_pad[i, j + \
                                                     dilation*(k-w)+pad, p],
                                    q[i, j, p]*k_pad[i, j+(k-w)+pad, p]), axis=p),
        name="G2BMM",
    )
    return [q, k, prob]


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def GBMML_compute(bs, n_heads, seq_len, feat_len, w, dilation, dtype):
    assert(feat_len % n_heads == 0)
    head_size = feat_len // n_heads
    prob = te.placeholder((bs*n_heads, seq_len, 2*w+1),
                          name="prob", dtype=dtype)
    q = te.placeholder((bs*n_heads, seq_len, head_size),
                       name="q", dtype=dtype)
    pad = dilation*w
    q_pad = te.compute(
        (bs*n_heads, seq_len+2*pad, head_size),
        lambda a, b, c: tvm.tir.if_then_else(
            tvm.tir.all(b >= pad, b - pad < seq_len),
            q[a, b - pad, c],
            tvm.tir.const(0.0, dtype),
        ),
        name="Qpad",
    )

    p = te.reduce_axis((0, 2*w+1), name="p")
    sum = te.compute(
        (bs*n_heads, seq_len, head_size),
        lambda i, j, k: te.sum(
            prob[i, j, p]*q_pad[i, j+dilation*(p-w)+pad, k], axis=p),
        name="GBMML",
    )
    return [prob, q, sum]

    ################################################################################
tasks = [
    tvm.auto_scheduler.SearchTask(
        func=longformer_compute0, args=(
            1, n_heads, seq_len, feat_len, w, 4, 8, dtype),
        target=target),
    tvm.auto_scheduler.SearchTask(
        func=longformer_compute0, args=(
            1, n_heads, seq_len, feat_len, w, 1, 8, dtype),
        target=target),
    # tvm.auto_scheduler.SearchTask(
    #     func=GBMML_compute, args=(
    #         1, n_heads, seq_len, feat_len, w, 1, dtype),
    #     target=target),
    # tvm.auto_scheduler.SearchTask(
    #     func=GBMML_compute, args=(
    #         1, n_heads, seq_len, feat_len, w, 4, dtype),
    #     target=target),
]

if tune:
    ################################################################################
    # Set Parameters for Auto-Scheduler
    tuner = auto_scheduler.TaskScheduler(tasks, strategy='round-robin')
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(
        timeout=20, min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=tuning_rounds*len(tasks),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        runner=measure_ctx.runner,
        verbose=2,
    )
    tuner.tune(tune_option)
else:  # Evaluation
    for task, task_name in zip(tasks, ('G2BMM_orig', 'G2BMM_opt')):
        inp, res = auto_scheduler.load_best_record(log_file, task.workload_key)

        # Rebuild the binary. This shows how you can apply the best schedule from a
        # log file without reruning the search again.
        ctx = tvm.cuda()
        sch, args = task.compute_dag.apply_steps_from_state(inp.state)
        func = tvm.build(sch, args, target)

        # print(tvm.lower(sch, args, simple_mode=True))
        # print("\n-------GPU code-------")
        # print(func.imported_modules[0].get_source())

        device = tvm.runtime.cuda()
        ctx = tvm.cuda()

        tvm_tensors = []
        bs = 1
        head_size = feat_len // n_heads
        shapes = [(bs*n_heads, seq_len, head_size), (bs*n_heads,
                                                     seq_len, head_size), (bs*n_heads, seq_len, 2*w+1)]
        for shape in shapes:
            a_np = np.random.uniform(size=shape).astype(np.float32)
            a_tvm = tvm.nd.array(a_np, device=device)
            tvm_tensors.append(a_tvm)
        evaluator = func.time_evaluator(func.entry_name, ctx, number=200)
        # evaluator(a_tvm, w_tvm, c_tvm)
        print(f"{task_name} Time: {evaluator(*tvm_tensors).mean*1000} ms")
