# Copyright (c) 2020 Software Platform Lab
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Software Platform Lab nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import argparse
from contextlib import closing

import pandas as pd
import torch
import numpy as np

from utils import get_inference_wrapper, evaluate, eval_result_to_df
from get_nnet_model import get_nnet_model, get_nnet_model_from_pth

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def main(args):
    # instantiate model and inputs
    model, dummy_input = get_nnet_model(args.model_name, args.bs)
    # print('dummy_input', dummy_input)

    with closing(get_inference_wrapper(model, dummy_input, args.mode)) as inference_wrapper:
        result = evaluate(inference_wrapper)
        # output = inference_wrapper.get_output()
        # print(np.shape(output))

    df = eval_result_to_df(args.mode, result)
    if args.out_path:
        df.to_csv(args.out_path)
    else:
        summary = pd.DataFrame({'mean (ms)': df.mean(), 'stdev (ms)': df.std()})
        print(summary)

    with closing(get_inference_wrapper(model, dummy_input, 'pytorch')) as inference_wrapper:
        result = evaluate(inference_wrapper)
        # output = inference_wrapper.get_output()
        # print(np.shape(output))

    df = eval_result_to_df(args.mode, result)
    if args.out_path:
        df.to_csv(args.out_path)
    else:
        summary = pd.DataFrame({'mean (ms)': df.mean(), 'stdev (ms)': df.std()})
        print(summary)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Model name')
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    parser.add_argument('--mode', type=str,
                        choices=['pytorch',
                                 'trace',
                                 'c2',
                                 'trt',
                                 'nimble',
                                 'nimble-multi'],
                        help='mode to conduct experiment')
    parser.add_argument('--out_path', type=str, default='', help='where to output the result')
    args = parser.parse_args()
    main(args)
    print(torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    print(torch.version.cuda, torch.backends.cudnn.is_available(), torch.backends.cudnn.version())
