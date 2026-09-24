"""Full pretrained ResMLP-12: native versus direct PTX affine/residual kernels."""
import argparse
import ctypes
import hashlib
import json
import time
import types
from pathlib import Path

import timm
import torch
from safetensors.torch import load_file
from timm.models.mlp_mixer import Affine

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=Path, required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
weight = args.weights
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.cuda.init()
model = timm.create_model('resmlp_12_224.fb_in1k', pretrained=False).eval().cuda()
model.load_state_dict(load_file(str(weight)))
x = torch.randn(2, 3, 224, 224, device='cuda')
lib = ctypes.CDLL(str(root / 'driver.so'))
lib.load_kernel.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
lib.load_kernel.restype = ctypes.c_int
lib.launch_affine.argtypes = [ctypes.c_void_p, *([ctypes.c_uint64] * 4), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
lib.launch_affine.restype = ctypes.c_int
function = ctypes.c_void_p()
assert lib.load_kernel(str(root / 'affine_fp32.cubin').encode(), b'affine_fp32', ctypes.byref(function)) == 0
tokens_function = ctypes.c_void_p()
assert lib.load_kernel(str(root / 'affine_tokens.cubin').encode(), b'affine_tokens', ctypes.byref(tokens_function)) == 0
residual_functions = {}
for name in ('residual_token', 'residual_channel'):
    handle = ctypes.c_void_p()
    assert lib.load_kernel(str(root / (name + '.cubin')).encode(), name.encode(), ctypes.byref(handle)) == 0
    residual_functions[name] = handle
residual_calls = 0


def residual(x, gamma, branch):
    global residual_calls
    assert x.stride() == (75264, 1, 196) and x.dtype == torch.float32
    token_order = branch.stride() == x.stride()
    assert token_order or branch.is_contiguous()
    kernel = residual_functions['residual_token' if token_order else 'residual_channel']
    out = torch.empty_like(x)
    code = lib.launch_affine(kernel, x.data_ptr(), out.data_ptr(), gamma.data_ptr(), branch.data_ptr(),
                             x.numel(), 384, torch.cuda.current_stream().cuda_stream)
    assert code == 0, code
    residual_calls += 1
    return out


def block_forward(self, x):
    branch = self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2)
    x = residual(x, self.ls1, branch)
    return residual(x, self.ls2, self.mlp_channels(self.norm2(x)))


block_originals = [b.forward for b in model.blocks]
assert all(isinstance(b.drop_path, torch.nn.Identity) for b in model.blocks)
calls = 0

def fused(self, value):
    global calls
    assert value.dtype == torch.float32 and value.shape[-1] == 384
    token_order = value.stride(-1) == 196 and value.stride(-2) == 1
    assert value.is_contiguous() or token_order
    kernel = tokens_function if token_order else function
    out = torch.empty_like(value)
    code = lib.launch_affine(kernel, value.data_ptr(), out.data_ptr(), self.alpha.data_ptr(), self.beta.data_ptr(),
                             value.numel(), 384, torch.cuda.current_stream().cuda_stream)
    assert code == 0, code
    calls += 1
    return out

norms = [m for m in model.modules() if isinstance(m, Affine)]
original = [m.forward for m in norms]

@torch.inference_mode()
def capture(optimized):
    global calls, residual_calls
    for block, method in zip(model.blocks, block_originals):
        block.forward = types.MethodType(block_forward, block) if optimized else method
    for layer, method in zip(norms, original):
        layer.forward = types.MethodType(fused, layer) if optimized else method
    for _ in range(5):
        result = model(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    calls = 0
    residual_calls = 0
    with torch.cuda.graph(graph):
        result = model(x)
    if optimized:
        assert calls == 25 and residual_calls == 24
    return graph, result

with torch.inference_mode():
    reference = model(x).clone()
    graphs, outputs = {}, {}
    for arm in ('native', 'ptx'):
        graphs[arm], outputs[arm] = capture(arm == 'ptx')
        graphs[arm].replay()
        torch.cuda.synchronize()
        assert torch.equal(outputs[arm], reference)
        if arm == 'ptx':
            assert torch.equal(outputs['ptx'], outputs['native'])
    rows = []
    orders = ['native', 'ptx', 'ptx', 'native'] if args.seed % 2 else ['ptx', 'native', 'native', 'ptx']
    for arm in orders:
        for _ in range(20):
            graphs[arm].replay()
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        for _ in range(100):
            graphs[arm].replay()
        stop.record()
        stop.synchronize()
        rows.append({'arm': arm, 'event_ms': start.elapsed_time(stop) / 100,
                     'wall_ms': (time.perf_counter() - wall) * 10})
    result = {'model': 'timm/resmlp_12_224.fb_in1k', 'seed': args.seed,
              'revision': '59a4112ca93db165fd828bef7d47d758ad84295b',
              'weight_sha256': hashlib.sha256(weight.read_bytes()).hexdigest(),
              'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(root.iterdir())
                                  if p.suffix in ('.ptx', '.cubin', '.so', '.cpp')},
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'torch': torch.__version__, 'timm': timm.__version__,
              'gpu': torch.cuda.get_device_name(), 'uuid': str(torch.cuda.get_device_properties(0).uuid),
              'mode': 'full CUDA graph replay', 'dtype': 'FP32', 'batch': 2,
              'affine_calls_per_forward': 25, 'residual_calls_per_forward': 24, 'bit_exact': True,
              'output_sha256': hashlib.sha256(reference.cpu().numpy().tobytes()).hexdigest(),
              'rows': rows}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
