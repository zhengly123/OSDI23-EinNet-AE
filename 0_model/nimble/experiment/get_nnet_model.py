from statistics import mode
import torch
import torch.nn as nn
import torchvision.models as models
from src.csrnet import csrnet
from src.dcgan import _netG as dcgan_netG
from src.cgan import GeneratorCGAN as cgan_netG
from src.GhostNet import ghost_net
from src.scnet import scnet101
from src.hetconv import vgg16bn as hetconv_vgg16bn 
from src.pyconv.build_model import build_model as pyconv_model
from src.pyconv.args_file import parser as pyconv_args
from src.DC_CDN_IJCAI21 import build_CDCN
from src.net48 import Net48
from src.testnet import TestNet
from src.gcn_1703_02719 import GCN
from src.unet import UNet
from src.srcnn import SRCNN
from src.infogan import infogan_G
from src.super_resolution.main import arch as WDSR # git@github.com:achie27/super-resolution.git
from src.longformer_torch import LongFormer

MODEL_PATH="/home/hsh/test-nimble/pytorch_model"

INPUT_SHAPES={
    "csrnet": (1, 512, 14, 14),
    "dcgan": (1, 100, 1, 1),
    "unet": (1, 3, 224, 224),
    "resnet18": (1, 3, 224, 224),
    "infogan": (1, 228, 1, 1),
    "gcn": (1, 3, 224, 224),
    "srcnn": (1, 1, 224, 224)
}

def _get_nnet_model(name, input_shape, precision=32):
    # fn = f"onnx/{name}.onnx"
    # dummy_input = torch.randn(*input_shape)
    dummy_input = torch.ones(*input_shape)
    print(name, '='*20)
    if name == 'csrnet':
        model = csrnet()
    elif name == 'dcgan':
        model = dcgan_netG(1)
    elif name == 'cgan':
        z_dim, c_dim = 100, 10
        model = cgan_netG(z_dim=z_dim, c_dim=c_dim)
    elif name == 'ghost':
        model = ghost_net()
    elif name == 'scnet':
        model = scnet101()
    elif name == 'hetconv':
        n_parts=1
        model = hetconv_vgg16bn(64, 64, 128, 128, 256, 256, 256,
                            512, 512, 512, 512, 512, 512, n_parts)
    elif name == 'pyconv': # Pyramid conv
        model = pyconv_model(pyconv_args.parse_args(args=['-a','pyconvresnet']))
    elif name =='dc-cdn':
        model = build_CDCN()
    elif name =='Net48':
        model = Net48()
    elif name =='TestNet':
        model = TestNet()
    elif name =='GCN':
        model = GCN(21)
    elif name =='unet':
        model = UNet(3, 10, [32, 64, 32], 0, 3, group_norm=0)
    elif name == "convTranspose":
        # model = nn.Sequential(nn.Conv2d(100, 512, 4, 1, 0, bias=False),nn.BatchNorm2d(512))
        model = nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False)
    elif name == "infogan":
        model = infogan_G()
    elif name in ['srcnn', 'fsrcnn', 'espcn', 'edsr', 'srgan', 'esrgan', 'prosr']:
        model = WDSR(name, 4, True).getModel()
    elif name == "longformer":
        model = LongFormer()
    else:
        model = getattr(models, name)(pretrained=True)

    if precision == 16:
        model = model.eval().cuda().half()
        dummy_input = dummy_input.cuda().half()
        # fn = f"onnx_fp16/{name}.onnx"
    else:
        model = model.eval().cuda()
        dummy_input = dummy_input.cuda()
    # # Without parameters
    # torch.onnx.export(model, dummy_input, fn,
    #   export_params=False, verbose=False)
    # torch.onnx.export(model, dummy_input, fn,
    #                   export_params=True, verbose=False, opset_version=13)
                    #   export_params=True, verbose=False, opset_version=12)
    # infer_onnx(fn)

    # Print model info
    # with open(f'info/{name}.txt', 'w') as f:
    #     print(model, file=f)
    
    return model, dummy_input


def get_nnet_model_from_pth(name, batch_size, precision=32):
    if name not in INPUT_SHAPES:
        raise f"model {name} is not supportted."
    print(name, '='*20)
    fn = f"{MODEL_PATH}/{name}.bs{batch_size}.onnx.pth"
    model = torch.load(fn)
    input_shape = (batch_size,) + INPUT_SHAPES[name][1:]
    dummy_input = torch.randn(*input_shape)
    if precision == 16:
        model = model.eval().cuda().half()
        dummy_input = dummy_input.cuda().half()
        # fn = f"onnx_fp16/{name}.onnx"
    else:
        model = model.eval().cuda()
        dummy_input = dummy_input.cuda()
    
    return model, dummy_input


def get_nnet_model(name, batch_size, precision=32):
    if name == "csrnet":
        input_shape = (batch_size, 512, 14, 14)
    elif name == "dcgan":
        input_shape = (batch_size, 100, 1, 1)
    elif name == "unet":
        input_shape = (batch_size, 3, 224, 224)
    elif name == "resnet18":
        input_shape = (batch_size, 3, 224, 224)
    elif name == "convTranspose":
        input_shape = (batch_size, 100, 1, 1)
    elif name == "infogan":
        input_shape = (batch_size, 228, 1, 1)
    elif name == "GCN":
        input_shape = (batch_size, 3, 224, 224)
    elif name in ['srcnn', 'fsrcnn', 'espcn', 'edsr', 'srgan', 'esrgan', 'prosr']:
        if name in ['srcnn', 'fsrcnn', 'espcn', 'srgan']:
            input_shape = (batch_size, 1, 32, 32)
        else:
            input_shape = (batch_size, 3, 32, 32)
    elif name == "longformer":
        input_shape = (batch_size, 10000, 512)
    else:
        raise f"Model {name} is not supported"
    
    return _get_nnet_model(name, input_shape, precision)

