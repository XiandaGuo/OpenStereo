import argparse
import contextlib
import datetime
import glob
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch

from easydict import EasyDict
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
MODEL_NAME = ''

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    RELATIVAE_ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from deploy_utils import (check_img_size, check_requirements, check_version, colorstr, file_size, get_default_args,
                          get_format_idx, load_model, print_args, url2file, yaml_save, Profile)
                     
MACOS = platform.system() == 'Darwin'  # macOS environment

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Network
sys.path.append(str(ROOT.parent))  # add ROOT to PATH
# from stereo.modeling import models, build_trainer
from stereo.utils.common_utils import config_loader, create_logger, load_params_from_file

from stereo.modeling.models.sttr.sttr import STTR
from stereo.modeling.models.psmnet.psmnet import PSMNet
from stereo.modeling.models.msnet.MSNet2D import MSNet2D
from stereo.modeling.models.msnet.MSNet3D import MSNet3D
from stereo.modeling.models.igev.igev_stereo import IGEVStereo as IGEV
from stereo.modeling.models.gwcnet.gwcnet import GwcNet
from stereo.modeling.models.fadnet.fadnet import FADNet
from stereo.modeling.models.coex.coex import CoEx
# from stereo.modeling.models.aanet.aanet import aanet
from stereo.modeling.models.cfnet.cfnet import CFNet
from stereo.modeling.models.casnet.cas_gwc import GwcNet as CasGwcNet
from stereo.modeling.models.casnet.cas_psm import PSMNet as CasPSMNet
from stereo.modeling.models.lightstereo.lightstereo import LightStereo as LightStereo
from stereo.modeling.models.stereobase.stereobase_gru import StereoBase as StereoBaseGRU


__net__ = {
    'STTR': STTR,
    'PSMNet': PSMNet,
    'MSNet2D': MSNet2D,
    'MSNet3D': MSNet3D,
    'IGEV': IGEV,
    'GwcNet': GwcNet,
    'FADNet': FADNet,
    'CoExNet': CoEx,
    # 'AANet': AANet,
    'CFNet': CFNet,
    'CasGwcNet': CasGwcNet,
    'CasPSMNet': CasPSMNet,
    'StereoBaseGRU': StereoBaseGRU,
    'LightStereo': LightStereo
}

# logger
logger = logging.getLogger('export')
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def export_formats():
    # export formats
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['TorchScript', 'torchscript', '.torchscript', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        # ['ONNX END2END', 'onnx_end2end', '_end2end.onnx', True, True],
        ['OpenVINO', 'openvino', '_openvino_model', True, False],
        ['TensorRT', 'engine', '.engine', False, True],
        ['CoreML', 'coreml', '.mlmodel', True, False],
        ['PaddlePaddle', 'paddle', '_paddle_model', True, True],]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])


def try_export(inner_func):
    # export decorator, i..e @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args['prefix']
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            logger.info(f'{prefix} export success 🍻 {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)')
            return f, model
        except Exception as e:
            logger.info(f'{prefix} export failure 😭 {dt.t:.1f}s: {e}')
            return None, None

    return outer_func


@try_export
def export_torchscript(model, inputs, file, optimize, prefix=colorstr('TorchScript:')):
    logger.info(f'\n{prefix} starting export with torch {torch.__version__}...')
    f = Path(file).with_suffix('.torchscript')

    ts = torch.jit.trace(model, inputs, strict=False)
    if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
        inputs['ref_img'].cpu()
        inputs['tgt_img'].cpu()
        model.cpu()
        optimize_for_mobile(ts)._save_for_lite_interpreter(str(f))
    else:
        ts.save(str(f))
    return f, None

@try_export
def export_onnx(model, inputs, weights, opset, dynamic, simplify, prefix=colorstr('ONNX:')):
    # ONNX export
    check_requirements('onnx', logger)
    import onnx

    logger.info(f'{prefix} starting export with onnx {onnx.__version__}...')
    f = Path(weights).with_suffix('.onnx')

    input_names = ['left_img', 'right_img']
    output_names =  ['disp_pred']

    if dynamic:
        dynamic = {'left_img': {0: 'batch', 2: 'height', 3: 'width'},
                   'right_img': {0: 'batch', 2: 'height', 3: 'width'}}

        dynamic['disp_pred'] = {1: 'height', 2: 'width'}

    
    torch.onnx.export(
        model,
        {'data': inputs},
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    onnx.save(model_onnx, f)

    # Simplify
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(('onnxruntime-gpu' if cuda else 'onnxruntime', 'onnx-simplifier>=0.4.1', 'onnxoptimizer'), logger)
        
            import onnxsim

            logger.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
            # model_opt, check = onnxsim.simplify(model_onnx, include_subgraph=True, skip_shape_inference=True)
            model_opt, check = onnxsim.simplify(model_onnx)
            logger.info("Here is the difference after simplification:")
            logger.info(onnxsim.model_info.print_simplifying_info(model_onnx, model_opt))
            assert check, 'assert check failed'
            onnx.save(model_opt, f)
        except Exception as e:
            logger.info(f'{prefix} simplifier failure: {e}')
    return f, model_opt if simplify else model_onnx

@try_export
def export_openvino(file, half, prefix=colorstr('OpenVINO:')):
    # OpenVINO export
    check_requirements('openvino-dev', logger)  # requires openvino-dev: https://pypi.org/project/openvino-dev/
    import openvino

    logger.info(f'{prefix} starting export with openvino {openvino.__version__}...')
    f = str(file).replace('.pt', f'_openvino_model{os.sep}')

    half_arg = '--compress_to_fp16'
    half_arg += '=True' if half else '=False'
    cmd = f"mo --input_model {Path(file).with_suffix('.onnx')} --output_dir {f} {half_arg}"
    subprocess.run(cmd.split(), check=True, env=os.environ)  # export
    return f, None

@try_export
def export_engine(model, inputs, file, half, dynamic, simplify, optimize, workspace=4, verbose=False, prefix=colorstr('TensorRT:')):
    # TensorRT export https://developer.nvidia.com/tensorrt
    assert inputs['left'].device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
    try:
        import tensorrt as trt
        import modelopt
    except Exception:
        if platform.system() == 'Linux':
            check_requirements('tensorrt', logger, cmds='-U --index-url https://pypi.ngc.nvidia.com')
            check_requirements('nvidia-modelopt[all]~=0.11.0', logger, cmds='--extra-index-url https://pypi.nvidia.com')
            # check_requirements('nvidia-modelopt[all]~=0.11.0', logger, cmds='-i https://pypi.tuna.tsinghua.edu.cn/simple')
        import tensorrt as trt
        import modelopt

    # TensorRT >= 8.4.0
    check_version(trt.__version__, '8.4.0', hard=True)  # require tensorrt>=8.4.0
    # export_onnx(model, inputs, file, 12, dynamic, simplify)  # opset 12
    onnx = Path(file).with_suffix('.onnx')

    logger.info(f'{prefix} starting export with TensorRT {trt.__version__}...')
    assert onnx.exists(), f'failed to export ONNX file: {onnx}'
    f = Path(file).with_suffix('.engine')  # TensorRT engine file
    trt_logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        trt_logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * 1 << 30)

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        logger.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        logger.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    logger.info(f'{prefix} building {colorstr("red", "FP16") if builder.platform_has_fast_fp16 and half else colorstr("red", "FP32")} engine as {f}')
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)

    with builder.build_serialized_network(network, config) as engine, open(f, 'wb') as plan:
        plan.write(engine)
    return f, None

@try_export
def export_paddle(model, inputs, file, prefix=colorstr('PaddlePaddle:')):
    # Paddle export
    check_requirements(('paddlepaddle', 'x2paddle'), logger, cmds='-i https://pypi.tuna.tsinghua.edu.cn/simple')
    import x2paddle
    from x2paddle.convert import pytorch2paddle

    logger.info(f'{prefix} starting export with X2Paddle {x2paddle.__version__}...')
    f = str(file).replace('.pt', f'_paddle_model{os.sep}')

    pytorch2paddle(module=model, save_dir=f, jit_type='trace', input_examples=[inputs])  # export
    return f, None


@try_export
def export_coreml(model, im, file, int8, half, prefix=colorstr('CoreML:')):
    # CoreML export
    check_requirements('coremltools', logger, cmds='-i https://pypi.tuna.tsinghua.edu.cn/simple')
    import coremltools as ct

    logger.info(f'{prefix} starting export with coremltools {ct.__version__}...')
    f = Path(file).with_suffix('.mlmodel')

    ts = torch.jit.trace(model, im, strict=False)  # TorchScript model
    ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=im['left'].shape, scale=1 / 255, bias=[0, 0, 0])])
    bits, mode = (8, 'kmeans_lut') if int8 else (16, 'linear') if half else (32, None)
    if bits < 32:
        if MACOS:  # quantization only supported on macOS
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress numpy==1.20 float warning
                ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
        else:
            print(f'{prefix} quantization only supported on macOS, skipping...')
    ct_model.save(f)
    return f, ct_model


def run(
        config=ROOT / '../../cfgs/psmnet/psmnet_kitti15.yaml',                                                                 # 'config.yaml path'
        weights=ROOT / '../../output/KITTI2015/PSMNet/PSMNet_SceneFlow/checkpoints/PSMNet_SceneFlow_epoch_latest.pt',           # weights path
        imgsz=(256, 512),                   # image (height, width)
        batch_size=1,                       # batch size
        device='cpu',                       # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=('torchscript', 'onnx'),    # include formats
        half=False,                         # FP16 half-precision export
        optimize=False,                     # TorchScript: optimize for mobile
        int8=False,                         # CoreML INT8 quantization
        dynamic=False,                      # ONNX/TensorRT: dynamic axes
        simplify=True,                      # ONNX: simplify model
        opset=17,                           # ONNX: opset version
        verbose=False,                      # TensorRT: verbose log
        workspace=4,                        # TensorRT: workspace size (GB)
):
    t = time.time()
    include = [x.lower() for x in include]  # to lowercase
    fmts = tuple(export_formats()['Argument'][1:])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f'ERROR: Invalid --include {include}, valid --include arguments are {fmts}'
    jit, onnx, xml, engine, coreml, paddle = flags  # export booleans
    file = Path(url2file(weights) if str(weights).startswith(('http:/', 'https:/')) else weights)  # PyTorch weights

    # Load PyTorch model
    if device != 'cpu':
        cuda_idx = -1
        try:
            cuda_idx = int(device)
        except ValueError:
            print(f"Error: '{device}' cannot be converted to an integer.")
            return
        assert torch.cuda.device_count() >= cuda_idx, 'cuda idx out of range'
        device = 'cuda:' + device
    if half:
        assert device != 'cpu' or coreml, '--half only compatible with GPU export, i.e. use --device 0'
        assert not dynamic, '--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both'

    # Network
    yaml_config = config_loader(opt.config)
    cfgs = EasyDict(yaml_config)
    model_name = cfgs.MODEL.NAME

    # Load pretrained model
    if not os.path.isfile(weights):
        raise FileNotFoundError
    output_model = load_model(__net__[model_name](cfgs.MODEL), weights)

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand

    # Input
    logger.info('imgsz: ' + str(imgsz))
    logger.info('please make sure that img_size are multiples of max stride')
    imgsz = [x for x in imgsz]
    left_img = torch.zeros(batch_size, 3, *imgsz, dtype=torch.float).to(device)
    right_img = torch.zeros(batch_size, 3, *imgsz, dtype=torch.float).to(device)
    inputs = {'left' : left_img, 'right' : right_img}
    
    # Prepare Model
    output_model.eval().to(device)
    if half and not coreml:
        # to FP16
        # TODO: complete forward using fp16
        # inputs['left'] = inputs['left'].half()
        # inputs['right'] = inputs['right'].half()
        # output_model = output_model.half().cuda()
        logger.info(colorstr('TODO: ') + 'complete forward using fp16, still using fp32')
    logger.info(f"starting from {file} ({file_size(file):.1f} MB)")

    # dry runs & check
    try:
        output = output_model(inputs)
        if not isinstance(output, (list, tuple, torch.Tensor)):
            raise TypeError(f"Expected a sequence type, but received {type(output)}")
        logger.info("Model output is a valid sequence type.")
    except TypeError as e:
        logger.warning(f"Error: {e}")

    # Exports
    f = [''] * len(fmts)  # exported filenames
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    
    fmts_df = export_formats()[1:]
    if jit:  # TorchScript
        f[get_format_idx(fmts_df, 'torchscript')], _ = export_torchscript(output_model, inputs, weights, optimize)
    if onnx or xml:  # OpenVINO requires ONNX
        f[get_format_idx(fmts_df, 'onnx')], _ = export_onnx(output_model, inputs, weights, opset, dynamic, simplify)
    if xml:  # OpenVINO
        f[get_format_idx(fmts_df, 'openvino')], _ = export_openvino(weights, half)
    if engine:  # TensorRT requires ONNX
        f[get_format_idx(fmts_df, 'engine')], _ = export_engine(output_model, inputs, weights, half, dynamic, simplify, optimize, workspace, verbose)
    if coreml:  # CoreML
        f[get_format_idx(fmts_df, 'coreml')], _ = export_coreml(output_model, inputs, weights, int8, half)
    if paddle:  # PaddlePaddle
        f[get_format_idx(fmts_df, 'paddle')], _ = export_paddle(output_model, inputs, weights)

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        logger.info(f'Export complete ({time.time() - t:.1f}s)'
                         f"\nResults saved to: {colorstr('bold', file.parent.resolve())}"
                         f"\ninclude types:    {colorstr('bold', include)}"
                         # f"\nValidate:        python {dir / 'val.py'} --weights {f[-1]} {h}"
                         f"\nVisualize:        https://netron.app")
    return f  # return list of exported files/dirs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=ROOT / '../../cfgs/psmnet/psmnet_kitti15.yaml', help='<config>.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / '../../output/KITTI2015/PSMNet/PSMNet_SceneFlow/checkpoints/PSMNet_SceneFlow_epoch_latest.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[256, 512], help='image size (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile / TensorRT: optimize inference')
    parser.add_argument('--int8', action='store_true', help='CoreML INT8 quantization')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TensorRT: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=17, help='ONNX: opset version')
    parser.add_argument('--verbose', action='store_true', help='TensorRT: verbose log'),
    parser.add_argument('--workspace', type=int, default=4, help='TensorRT: workspace size (GB)')
    parser.add_argument(
        '--include',
        nargs='+',
        default=['torchscript', 'onnx'],
        help='torchscript, onnx, openvino, engine, coreml, paddle')
    opt = parser.parse_args()

    print_args(vars(opt))
    return opt


def main(opt):
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
