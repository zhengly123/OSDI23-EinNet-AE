import os
import sys

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
log_filename = "conv5x5.ansor.log"

np.random.seed(0)


@auto_scheduler.register_workload
def op_batch_matmul(B, M, N, K):
    A = te.placeholder((B, M, K), name="A")
    B = te.placeholder((B, N, K), name="B")
    out = topi.nn.batch_matmul(A, B)
    return [A, B, out]


@auto_scheduler.register_workload
def op_conv2D(n, c, h, w, f, r, s, padding, stride, dilation):
    data = te.placeholder((n, c, h, w), name="data")
    kernel = te.placeholder((f, c, r, s), name="kernel")
    # bias = te.placeholder((1, CO, 1, 1), name="bias")
    out = topi.nn.conv2d_nchw(
        data, kernel, stride, padding, dilation=dilation, out_dtype=dtype)
    return [data, kernel, out]


@auto_scheduler.register_workload
def op_conv2D_winograd(n, c, h, w, f, r, s, padding, stride, dilation):
    data = te.placeholder((n, h, w, c), name="data")
    kernel = te.placeholder((r, s, c, f), name="kernel")
    # bias = te.placeholder((1, CO, 1, 1), name="bias")
    out = topi.nn.conv2d_winograd_nhwc(
        data, kernel, stride, padding, dilation, out_dtype=dtype)
    return [data, kernel, out]


@auto_scheduler.register_workload
def op_conv2D_transpose(n, c, h, w, f, r, s, padding, stride, dilation, output_padding=(0, 0)):
    assert(dilation == 1)
    data = te.placeholder((n, f, h, w), name="data")
    kernel = te.placeholder((f, c, r, s), name="kernel")
    out = topi.nn.conv2d_transpose_nchw(data, kernel, strides=(
        stride, stride), padding=padding, out_dtype=dtype, output_padding=output_padding)
    return [data, kernel, out]


def ansor_tune(tasks):
    # Inspect the computational graph
    # print(task.compute_dag)

    ######################################################################
    # * :code:`measure_ctx` launches a different process for measurement to
    #   provide isolation. It can protect the master process from GPU crashes
    #   during measurement and avoid other runtime conflicts.
    # * :code:`min_repeat_ms` defines the minimum duration of one "repeat" in every measurement.
    #   This can warmup the GPU, which is necessary to get accurate measurement results.
    #   Typically, we recommend a value > 300 ms.
    # * :code:`num_measure_trials` is the number of measurement trials we can use during the search.
    #   We only make 10 trials in this tutorial for a fast demonstration. In practice, 1000 is a
    #   good value for the search to converge. You can do more trials according to your time budget.
    # * In addition, we use :code:`RecordToFile` to dump measurement records into a file `conv2d.json`.
    #   The measurement records can be used to query the history best, resume the search,
    #   and do more analyses later.
    # * see :any:`auto_scheduler.TuningOptions`,
    #   :any:`auto_scheduler.LocalRPCMeasureContext` for more parameters.

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        # change this to 1000 to achieve the best performance
        num_measure_trials=1024*len(tasks)+1,
        runner=measure_ctx.runner,
        # runner=auto_scheduler.RPCRunner(
        #     "nico2_v100_32",  # change the device key to your key
        #     "0.0.0.0",
        #     9190,
        #     n_parallel=7,
        #     number=5,
        #     repeat=1,
        #     timeout=20,
        #     min_repeat_ms=300,
        # ),
        measure_callbacks=[auto_scheduler.RecordToFile(log_filename)],
        verbose=2,
    )

    print("Begin tuning...")
    tuner = auto_scheduler.TaskScheduler(tasks, strategy='round-robin')
    tuner.tune(tune_option)


def ansor_load(task):
    # print(task.workload_key, len(task.workload_key[1:]))
    n, c, h, w, f, r, s, padding, stride, dilation = eval(task.workload_key)[
        1:]
    print(eval(task.workload_key)[1:])
    inp, res = auto_scheduler.load_best_record(log_filename, task.workload_key)

    # # Print equivalent python schedule API. This can be used for debugging and
    # # learning the behavior of the auto-scheduler.
    # print("Equivalent python schedule:")
    # print(task.compute_dag.print_python_code_from_state(inp.state))

    # Rebuild the binary. This shows how you can apply the best schedule from a
    # log file without reruning the search again.
    ctx = tvm.gpu()
    sch, args = task.compute_dag.apply_steps_from_state(inp.state)
    func = tvm.build(sch, args, target)

    # print(tvm.lower(sch, args, simple_mode=True))
    # print("\n-------GPU code-------")
    # print(func.imported_modules[0].get_source())

    # check correctness
    a_np = np.random.uniform(size=(n, c, h, w)).astype(np.float32)
    w_np = np.random.uniform(size=(f, c, r, s)).astype(np.float32)
    c_np = conv2d_nchw_python(a_np, w_np, stride, padding)
    print(a_np.shape, w_np.shape, c_np.shape)

    ctx = tvm.gpu()
    a_tvm = tvm.nd.array(a_np)
    w_tvm = tvm.nd.array(w_np)
    c_tvm = tvm.nd.empty(c_np.shape)
    func(a_tvm, w_tvm, c_tvm)
    tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

    evaluator = func.time_evaluator(func.entry_name, ctx, number=200)
    # evaluator(a_tvm, w_tvm, c_tvm)
    print(f"Time: {evaluator(a_tvm, w_tvm, c_tvm).mean*1000} ms")

    # ######################################################################
    # # A more complicated example is to resume the search.
    # # In this case, we need to create the search policy and cost model by ourselves
    # # and resume the status of search policy and cost model with the log file.
    # # In the example below we resume the status and do more 5 trials.

    # cost_model = auto_scheduler.XGBModel()
    # cost_model.update_from_file(log_file)
    # search_policy = auto_scheduler.SketchPolicy(
    #     task, cost_model, init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
    # )
    # measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    # tune_option = auto_scheduler.TuningOptions(
    #     num_measure_trials=5,
    #     runner=measure_ctx.runner,
    #     measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    # )
    # sch, args = auto_scheduler.auto_schedule(task, search_policy, tuning_options=tune_option)
    # Kill the measurement process
    # del measure_ctx


def ansor_dump_all_logs(task):
    # print(task.workload_key, len(task.workload_key[1:]))
    n, c, h, w, f, r, s, padding, stride, dilation = eval(task.workload_key)[
        1:]
    # inp, res = auto_scheduler.load_best(log_filename, task.workload_key)
    # logs = auto_scheduler.load_records(log_filename)

    log_reader = auto_scheduler.RecordReader(log_filename)
    best_cost = 1e30
    best_inp = None
    best_res = None

    logs = {}
    data = []
    cnt = 0
    for logid, (inp, res) in enumerate(log_reader):
        if res.error_no != auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
            continue
        if inp.task.workload_key != task.workload_key:
            continue
        if target and inp.task.target.kind.name != 'cuda':
            continue

        # # dump schedule
        # if logid in (7755, 7946):
        #     sch, args = task.compute_dag.apply_steps_from_state(inp.state)
        #     print(tvm.lower(sch, args, simple_mode=True))
        if logid not in (7494, 7296):
            continue

        sch, args = task.compute_dag.apply_steps_from_state(inp.state)
        lowered_schedule = str(tvm.lower(sch, args, simple_mode=True))
        searched = re.search(
            'IterVar\(blockIdx.x:.*thread_extent" = (\d*)', lowered_schedule)
        grid_size = int(searched.group(1))
        searched = re.search(
            'IterVar\(threadIdx.x:.*thread_extent" = (\d*)', lowered_schedule)
        block_size = int(searched.group(1))
        if grid_size not in logs:
            logs[grid_size] = []
        logs[grid_size].append(lowered_schedule)
        # cnt += 1
        # if cnt % 25 == 0:
        #     print(cnt)

        costs = [v.value for v in res.costs]
        cost = np.mean(costs)
        if cost < best_cost:
            best_cost = cost
            best_inp = inp
            best_res = res
        data.append((logid, grid_size, block_size, cost))

    # print(data, file=open('squeezenet.ansor.analysis.3x3.txt', 'w'))
    # return

    # Print equivalent python schedule API. This can be used for debugging and
    # learning the behavior of the auto-scheduler.
    # print("Equivalent python schedule:")
    print(task.compute_dag.print_python_code_from_state(best_inp.state))

    breakpoint()
    ctx = tvm.gpu()
    sch, args = task.compute_dag.apply_steps_from_state(best_inp.state)
    print(tvm.lower(sch, args, simple_mode=True))

    func = tvm.build(sch, args, target)
    print("\n\n\n-------GPU code-------")
    print(func.imported_modules[0].get_source())


def conv2d_nchw_fake_kernel_size(Input, Filter, stride, padding, dilation, out_dtype=None):
    from tvm.topi.utils import simplify
    """Convolution operator in NCHW layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    # pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
    #     padding, (dilated_kernel_h, dilated_kernel_w)
    # )
    pad_top, pad_left, pad_down, pad_right = 1, 1, 1, 1
    out_channel = num_filter
    out_height = simplify(
        (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w +
                         pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = topi.nn.pad(Input, pad_before, pad_after, name="pad_temp")
    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    return te.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: te.sum(
            te.if_then_else(xx == 0 and yy == 0,
                            temp[nn, rc, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w].astype(
                                out_dtype
                            )
                            * Filter[ff, rc, ry, rx].astype(out_dtype), 0),
            axis=[rc, ry, rx],
        ),
        tag="conv2d_nchw",
    )

    # print(tvm.lower(s, args, simple_mode=True))
    # func = tvm.build(s, args, target)
    # print("\n\n\n-------GPU code-------")
    # print(func.imported_modules[0].get_source())


task_parameters = [
    # Conv (n, c, h, w, f, r, s, padding, stride, dilation),
    # Conv3x3 -> Gemm
    (op_conv2D, (1, 512, 7, 7, 512, 3, 3, 1, 1, 1), "Conv3x3_origin"),
    (op_batch_matmul, (1, 1*7*7, 512*3*3, 512), "Conv3x3_opt"),

    # ConvTranpose -> Gemm
    (op_conv2D_transpose, (16, 256, 2, 2, 448, 4, 4, 1, 2, 1), "ConvTranspose"),
    (op_batch_matmul, (1, 64, 4096, 448), "ConvTranspose_opt"),

    # Conv5x5 -> Conv3x3
    (op_conv2D, (16, 32, 224, 224, 1, 5, 5, 2, 1, 1), "Conv5x5_origin"),
    (op_conv2D, (16, 32, 224, 224, 4, 3, 3, 2, 1, 1), "Conv5x5_opt"),
    (op_conv2D_winograd, (16, 32, 224, 224, 4, 3, 3,
     2, 1, 1), "Conv5x5_opt_winograd"),  # Winograd
]

tasks = []
for fn, params, desc in task_parameters:
    print(params)
    tasks.append(auto_scheduler.SearchTask(
        fn, params, target=target, desc=desc))
print('# of tasks = %d' % (len(tasks)))

# # Tuning
# ansor_tune(tasks)

# Evaluating
for task in tasks:
    ansor_load(task)
#     # ansor_dump_all_logs(task)
    print('\n'*5)

# Customize the topi schedule (disable loop unroll)
# for params in task_parameters:
#     mock_ansor_schedule(*params)
