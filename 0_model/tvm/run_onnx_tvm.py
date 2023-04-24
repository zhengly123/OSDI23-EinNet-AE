import onnx
from onnx import TensorProto
import numpy as np
import os
import tvm
from tvm import te
import tvm.relay as relay
from tvm import relay, autotvm
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_executor as runtime
from tvm import te, auto_scheduler
from tvm.contrib import graph_executor
import sys
import argparse
# import logging
# logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('backend', type=str)
parser.add_argument('target', type=str)
args = parser.parse_args()

def get_shape(tensor):
    onnx_dtype = tensor.type
    shape = []
    for s in onnx_dtype.tensor_type.shape.dim:
        shape.append(s.dim_value)
    return shape


def get_io_shape(onnx_model):
    input_shape = [get_shape(t) for t in onnx_model.graph.input]
    output_shape = [get_shape(t) for t in onnx_model.graph.output]
    return input_shape, output_shape


def get_network(model_name):
    onnx_model = onnx.load(f"{model_name}")
    input_shape, output_shape = get_io_shape(onnx_model)
    mod, params = relay.frontend.from_onnx(onnx_model)
    return mod, params, input_shape, output_shape, onnx_model


def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log", # !!!
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    # os.remove(tmp_log_file)


def get_numpy(tensor):
    # ONNX Data Types Doc: https://github.com/onnx/onnx/blob/master/docs/IR.md#standard-data-types
    # ONNX Data Types Code: https://github.com/onnx/onnx/blob/master/onnx/defs/data_type_utils.h
    # NumPy Data Types: https://numpy.org/doc/stable/user/basics.types.html

    def onnx2np(dtype):
        # https://www.javadoc.io/static/org.nd4j/nd4j-api/1.0.0-beta5/onnx/Onnx.TensorProto.DataType.html
        if dtype == TensorProto.BOOL: return np.bool_
        if dtype == TensorProto.FLOAT: return np.float32
        if dtype == TensorProto.INT64: return np.int64
        raise NotImplementedError(dtype)
    # print(dir(tensor.type.tensor_type), tensor.type.tensor_type, type(tensor.type.tensor_type))
    dtype = onnx2np(tensor.type.tensor_type.elem_type)
    shape = []
    for dim in tensor.type.tensor_type.shape.dim:
        shape.append(dim.dim_value)
    # shape = tensor.type.tensor_type.shape
    return np.ones(shape, dtype=dtype)


def tune_and_evaluate_ansor(model_name, target, n_trials, log_file):
    print("Extract tasks...")
    mod, params, input_shape, output_shape, onnx_model = get_network(model_name=model_name)
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    if len(tasks) > 0: 
        for idx, task in enumerate(tasks):
            print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
            print(task.compute_dag)

        print("Begin tuning...", flush=True)
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials * len(tasks),
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

        tuner.tune(tune_option)
    else:
        print('No tuning task')

    # Compile with the history best
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)

    # Create graph executor
    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    data_tvm = {t.name: tvm.nd.array(get_numpy(t)) for t in onnx_model.graph.input} # !!!!!!!
    module.set_input(**data_tvm)

    # Evaluate
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, repeat=1, number=100))
    # profile_start()
    print(module.benchmark(dev, repeat=1, number=100)) # and record the min value
    # profile_stop()
    # print(module.benchmark(dev, repeat=3, min_repeat_ms=500)) # the min value


def tune_and_evaluate(tuning_opt, model_name, target):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape, out_shape, onnx_model = get_network(model_name=model_name)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"), relay.op.get("nn.dense"), relay.op.get("nn.batch_matmul"))
    )

    # run tuning tasks
    print("Tuning...")
    # tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(tuning_opt['log_filename']):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

        # load parameters
        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))
        data_tvm = {t.name: tvm.nd.array(get_numpy(t)) for t in onnx_model.graph.input}
        # data_tvm = tuple( for x in input_shape)
        module.set_input(**data_tvm)

        # evaluate
        print("Evaluate inference time cost...")
        print(module.benchmark(dev, number=1, repeat=600))


def tune(model_path, backend, device):
    print(f"model_path={model_path} on {backend}")
    assert device in ['sm_70', 'sm_80']
    if backend=='ansor':
        target=f"cuda -arch={device}"
    elif backend=='cublas':
        target=f"cuda -libs=cudnn,cublas -arch={device}"
    else:
        assert False
    model_name = model_path.split('/')[-1].strip()
    log_file = f"{model_name}_{backend}_{device}/tvm.log"
    dtype = "float32"

    tuning_option = {
        "log_filename": log_file,
        "tuner": "xgb",
        "n_trial": 1000,
        "early_stopping": 600,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        ),
    }
    # target = tvm.target.cuda()
    tune_and_evaluate_ansor(model_path, target, tuning_option['n_trial'], tuning_option['log_filename'])


tune(args.model, args.backend, args.target)
