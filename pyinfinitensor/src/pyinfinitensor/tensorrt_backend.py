import subprocess
import re
import os
from .onnx import save_onnx


def get_trt_time(g):
    onnx_filename = './tmp.onnx'
    save_onnx(g, onnx_filename)
    plugin_path = os.environ['TRT_PLUGIN_LIB']
    cmd = f'trtexec --noTF32 --onnx={onnx_filename} --plugins={plugin_path}/libnvinfer_plugin.so.8.2.0'
    print(cmd)
    res = subprocess.run(cmd.split(' '), capture_output=True)
    p = re.compile('GPU Compute Time.*mean = ([0-9.]+) ms')
    output = res.stdout.decode('utf-8')
    err = res.stderr.decode('utf-8')
    print(output, '\n'*5, err)
    return float(p.search(output).group(1))
