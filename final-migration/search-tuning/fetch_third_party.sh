#!/usr/bin/env bash
# 取回 CUDA 构建所需的 third_party（cutlass + TileLang）。
# 为什么需要：cpp/CMakeLists.txt:74-81 在 USE_BACKEND=CUDA 下对这两个目录做硬检查，
# 缺任一即 FATAL_ERROR。而 2026-10-03 的清理没有保留 third_party（体积大、可重取）。
#
# cutlass 版本选择依据：cpp/neuralnet/cudabackend_sm89_{dual,linear2,qkv_rope}_gemm.cu
# 用的是 CUTLASS **2.x 风格** threadblock API（cutlass/gemm/threadblock/mma_multistage.h、
# cutlass/gemm/device/gemm_batched.h）。CUTLASS 4.x 已删除该 API ⇒ 必须用 3.x。
# v3.4.1 同时保留 2.x API 与 examples/45_dual_gemm（flash-attn AOT 用，本构建不需要但无害）。
set -Eeuo pipefail

ROOT="${1:-/root/autodl-tmp/katago-plan/katago-src}"
# ⚠️ 不要用 tag，必须用源码里钉死的 commit。
# cpp/neuralnet/cudabackend_sm89_{dual,linear2,qkv_rope}_gemm.cu 的文件头都写着
#   "CUTLASS commit: 7127592069c2fe01b041e174ba4345ef9b279671"
# 实测教训：v3.4.1（3.x）会在 cpp/neuralnet/sm120_aot/outer_projection.cu 上报
#   error: type "cutlass::gemm::kernel::GemmUniversal<...>::Mma" ... enable_if_t<...>
# 因为 3.x 改了 MmaMultistage 的模板签名，2.x 风格的实例化不再成立。
CUTLASS_COMMIT="${CUTLASS_COMMIT:-7127592069c2fe01b041e174ba4345ef9b279671}"

# AutoDL 直连 GitHub 不通，必须走学术加速；env 不跨 ssh 会话，所以在这里 source。
if [ -f /etc/network_turbo ]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi

cd "$ROOT"
mkdir -p third_party
cd third_party

if [ ! -f cutlass/include/cutlass/cutlass.h ] || \
   [ "$(git -C cutlass rev-parse HEAD 2>/dev/null)" != "$CUTLASS_COMMIT" ]; then
  echo "== fetch cutlass @ $CUTLASS_COMMIT =="
  rm -rf cutlass
  mkdir -p cutlass
  git -C cutlass init -q
  git -C cutlass remote add origin https://github.com/NVIDIA/cutlass.git
  git -C cutlass fetch -q --depth 1 origin "$CUTLASS_COMMIT"
  git -C cutlass checkout -q FETCH_HEAD
else
  echo "== cutlass 已是目标 commit，跳过 =="
fi

if [ ! -d TileLang/.git ]; then
  echo "== clone TileLang（不取子模块，只需 src/tl_templates 存在） =="
  git clone --depth 1 https://github.com/tile-ai/TileLang.git TileLang 2>&1 | tail -5
else
  echo "== TileLang 已存在，跳过 =="
fi

echo "== 校验 =="
check() { if [ -e "$2" ]; then echo "  OK   $1"; else echo "  FAIL $1  ($2)"; fi; }
echo "  cutlass HEAD = $(git -C cutlass rev-parse HEAD 2>/dev/null)"
check "cutlass/cutlass.h"                  cutlass/include/cutlass/cutlass.h
check "cutlass mma_multistage.h"           cutlass/include/cutlass/gemm/threadblock/mma_multistage.h
check "cutlass device/gemm_batched.h"      cutlass/include/cutlass/gemm/device/gemm_batched.h
check "cutlass threadblock_swizzle.h"      cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h
check "cutlass 45_dual_gemm"               cutlass/examples/45_dual_gemm/device/dual_gemm.h
check "TileLang tl_templates/debug.h"      TileLang/src/tl_templates/cuda/debug.h
du -sh cutlass TileLang 2>/dev/null
echo "== done =="
