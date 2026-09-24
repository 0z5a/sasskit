#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
g++ -O3 -fPIC -shared -I"$CUDA_ROOT/include" driver.cpp -lcuda -o driver.so
for name in affine_fp32 affine_tokens residual_token residual_channel chain_token chain_channel; do
    "$CUDA_ROOT/bin/ptxas" -arch=sm_120 "$name.ptx" -o "$name.cubin"
done
