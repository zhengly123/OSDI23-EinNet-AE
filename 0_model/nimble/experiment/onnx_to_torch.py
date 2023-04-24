import onnx
import torch
from onnx2torch import convert
from onnx2pytorch import ConvertModel
import sys

onnx_model = onnx.load(sys.argv[1])
pytorch_model = convert(onnx_model)

pytorch_model = pytorch_model.cuda().eval()

torch.save(pytorch_model, sys.argv[2], _use_new_zipfile_serialization=False)
