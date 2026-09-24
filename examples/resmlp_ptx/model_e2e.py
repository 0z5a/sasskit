"""Full pretrained ResMLP-12: native, PTX residuals, and residual/affine fusion."""
import argparse
import ctypes
import hashlib
import json
import os
import sys
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
parser.add_argument('--validate-only', action='store_true', help='Run correctness checks without timing.')
args = parser.parse_args()
root = Path(__file__).resolve().parent
weight = args.weights
weight_sha256 = hashlib.sha256(weight.read_bytes()).hexdigest()
assert weight_sha256 == "2ed399f6a787a30925f49bd3e8040e6b0f0a620ed68d56758c703e98bb7c0fbe"
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
lib.launch_chain.argtypes = [ctypes.c_void_p, *([ctypes.c_uint64] * 7), ctypes.c_uint32, ctypes.c_void_p]
lib.launch_chain.restype = ctypes.c_int
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
chain_functions = {}
for name in ('chain_token', 'chain_channel'):
    handle = ctypes.c_void_p()
    assert lib.load_kernel(str(root / (name + '.cubin')).encode(), name.encode(), ctypes.byref(handle)) == 0
    chain_functions[name] = handle
chain_calls = 0


def residual_affine(value: torch.Tensor, gamma: torch.Tensor, branch: torch.Tensor,
                    norm: Affine) -> tuple[torch.Tensor, torch.Tensor]:
    global chain_calls
    assert value.shape[1:] == (196, 384) and value.stride() == (75264, 1, 196)
    assert value.dtype == branch.dtype == torch.float32 and branch.shape == value.shape
    token_order = branch.stride() == value.stride()
    assert token_order or branch.is_contiguous()
    kernel = chain_functions['chain_token' if token_order else 'chain_channel']
    out, normalized = torch.empty_like(value), torch.empty_like(value)
    code = lib.launch_chain(kernel, value.data_ptr(), out.data_ptr(), normalized.data_ptr(),
                            gamma.data_ptr(), branch.data_ptr(), norm.alpha.data_ptr(), norm.beta.data_ptr(),
                            value.numel(), torch.cuda.current_stream().cuda_stream)
    assert code == 0, code
    chain_calls += 1
    return out, normalized


def features_chain(self, value: torch.Tensor) -> torch.Tensor:
    value = self.stem(value)
    normalized = self.blocks[0].norm1(value)
    for i, block in enumerate(self.blocks):
        branch = block.linear_tokens(normalized.transpose(1, 2)).transpose(1, 2)
        value, normalized = residual_affine(value, block.ls1, branch, block.norm2)
        branch = block.mlp_channels(normalized)
        next_norm = self.blocks[i + 1].norm1 if i + 1 < len(self.blocks) else self.norm
        value, normalized = residual_affine(value, block.ls2, branch, next_norm)
    return normalized


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
features_original = model.forward_features
assert all(isinstance(b.drop_path, torch.nn.Identity) for b in model.blocks)
assert not model.grad_checkpointing and len(model.blocks) == 12
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
def capture(arm):
    global calls, residual_calls, chain_calls
    model.forward_features = types.MethodType(features_chain, model) if arm == 'chain' else features_original
    for block, method in zip(model.blocks, block_originals):
        block.forward = types.MethodType(block_forward, block) if arm == 'ptx' else method
    for layer, method in zip(norms, original):
        layer.forward = types.MethodType(fused, layer) if arm != 'native' else method
    for _ in range(5):
        result = model(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    calls = 0
    residual_calls = 0
    chain_calls = 0
    with torch.cuda.graph(graph):
        result = model(x)
    assert (calls, residual_calls, chain_calls) == {
        'native': (0, 0, 0), 'ptx': (25, 24, 0), 'chain': (1, 0, 24)
    }[arm]
    return graph, result


def assert_equal_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


with torch.inference_mode():
    # Odd batches exercise the half-full final CTA; both branch layouts must agree.
    kernel_cases = 0
    for batch in (1, 3):
        for token_order in (False, True):
            for scale in (0.0, 16.0):
                value = torch.randn(batch, 384, 196, device='cuda').transpose(1, 2) * scale
                branch = torch.randn_like(value) if token_order else torch.randn(batch, 196, 384, device='cuda')
                block = model.blocks[0]
                expected = value + block.ls1 * branch
                actual, normalized = residual_affine(value, block.ls1, branch, block.norm2)
                assert_equal_bits(actual, expected)
                assert_equal_bits(normalized, block.norm2(expected))
                kernel_cases += 1
    # Check every fused boundary against the original modules before graph capture.
    value = model.stem(x)
    for i, block in enumerate(model.blocks):
        branch = block.linear_tokens(block.norm1(value).transpose(1, 2)).transpose(1, 2)
        expected = value + block.ls1 * branch
        value, normalized = residual_affine(value, block.ls1, branch, block.norm2)
        assert_equal_bits(value, expected)
        assert_equal_bits(normalized, block.norm2(expected))
        branch = block.mlp_channels(normalized)
        expected = value + block.ls2 * branch
        norm = model.blocks[i + 1].norm1 if i < 11 else model.norm
        value, normalized = residual_affine(value, block.ls2, branch, norm)
        assert_equal_bits(value, expected)
        assert_equal_bits(normalized, norm(expected))
    reference = model(x).clone()
    assert torch.isfinite(reference).all()
    graphs, outputs = {}, {}
    for arm in ('native', 'ptx', 'chain'):
        graphs[arm], outputs[arm] = capture(arm)
        graphs[arm].replay()
        torch.cuda.synchronize()
        assert_equal_bits(outputs[arm], reference)
    rows = []
    arms = ['native', 'ptx', 'chain']
    offset = args.seed % 3
    arms = arms[offset:] + arms[:offset]
    orders = [] if args.validate_only else arms + arms[::-1]
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
              'weight_sha256': weight_sha256,
              'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(root.iterdir())
                                  if p.suffix in ('.ptx', '.cubin', '.so', '.cpp')},
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'torch': torch.__version__, 'timm': timm.__version__,
              'pid': os.getpid(), 'source_root': str(root), 'python': sys.executable, 'timm_path': timm.__file__,
              'input_sha256': hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
              'gpu': torch.cuda.get_device_name(), 'uuid': str(torch.cuda.get_device_properties(0).uuid),
              'mode': 'correctness only' if args.validate_only else 'full CUDA graph replay',
              'dtype': 'FP32', 'batch': 2,
              'ptx_calls_per_forward': {'native': 0, 'ptx': 49, 'chain': 25},
              'kernel_cases_checked': kernel_cases, 'fused_boundaries_checked': 24, 'bit_exact': True,
              'output_sha256': hashlib.sha256(reference.cpu().numpy().tobytes()).hexdigest(),
              'rows': rows}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
