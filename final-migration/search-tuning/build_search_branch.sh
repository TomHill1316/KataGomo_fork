#!/usr/bin/env bash
# 在 SM89 主机上编译 search-adaptive-explore 分支（新增 cpuctExplorationFloorCoeff /
# rootMinVisitShare 两个参数）。
#
# 设计要点
#   * 输出到独立 build 目录，**不覆盖** bin/katago-prune-sched-grace（生产参照二进制）。
#   * -DKATAGO_CUDA_ARCHITECTURES=89：只生成 sm_89，显著缩短编译时间。
#     （SM120 的那几个 stub 源在 CMakeLists 里被显式钉成 arch 120，体积很小，不影响。）
#   * 并发度保守：CUTLASS 的 nvcc TU 单个可达数 GB，cgroup 内存上限 64 GB。
#     J 太大 → OOM kill（本机踩过：无卡模式 2 GiB 时连 -fsyntax-only 都 OOM）。
set -Eeuo pipefail

ROOT="${ROOT:-/root/autodl-tmp/katago-plan/katago-src}"
BUILD="${BUILD:-/root/autodl-tmp/katago-plan/katago-build-sm89}"
J="${J:-16}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:/root/miniconda3/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

command -v nvcc >/dev/null || { echo "nvcc 不在 PATH（无卡模式？）"; exit 1; }
command -v ninja >/dev/null || { echo "ninja 不在 PATH"; exit 1; }

for p in "${ROOT}/third_party/cutlass/include/cutlass/cutlass.h" \
         "${ROOT}/third_party/TileLang/src/tl_templates/cuda/debug.h"; do
  [ -e "$p" ] || { echo "缺 third_party 依赖: $p —— 先跑 tools/fetch_third_party.sh"; exit 1; }
done

# SM89 AOT 内核（flash-attn / dual-gemm / linear2 / preConv / postConv / qkv-rope）。
# 缺了它 → 主 target 拿不到 KATAGO_ENABLE_SM89_* 宏 → 生产 plan 要求的 tactic 全部
# fail-closed，GTP 直接起不来。先跑 tools/fetch_flash_attn.sh。
FA_ROOT="${FA_ROOT:-${ROOT}/third_party/flash-attention}"
[ -f "${FA_ROOT}/hopper/flash_fwd_launch_template.h" ] || {
  echo "缺 SM89 flash-attention: ${FA_ROOT} —— 先跑 tools/fetch_flash_attn.sh"; exit 1; }
grep -q "KATAGO_FLASH_BLOCK_N" "${FA_ROOT}/hopper/tile_size.h" || {
  echo "flash-attention-sm89.patch 未应用（tile_size.h 无 KATAGO_FLASH_BLOCK_N）"; exit 1; }

multiarch="$(gcc -print-multiarch)"
zlib="/usr/lib/${multiarch}/libz.so"
[ -r "$zlib" ] || { echo "缺系统 zlib: $zlib"; exit 1; }

echo "=== configure ==="
cmake -S "${ROOT}/cpp" -B "${BUILD}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_BACKEND=CUDA \
  -DBUILD_DISTRIBUTED=0 \
  -DUSE_TCMALLOC=0 \
  "-DCMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc" \
  -DCUDNN_ROOT_DIR=/usr \
  -DKATAGO_CUDA_ARCHITECTURES=89 \
  "-DKATAGO_TILELANG_ROOT=${ROOT}/third_party/TileLang" \
  "-DKATAGO_CUTLASS_ROOT=${ROOT}/third_party/cutlass" \
  "-DSM89_FLASH_ATTN_ROOT=${FA_ROOT}" \
  -DZLIB_INCLUDE_DIR=/usr/include \
  "-DZLIB_LIBRARY=${zlib}" \
  "-DZLIB_LIBRARY_RELEASE=${zlib}"

echo "=== build (-j${J}) ==="
cmake --build "${BUILD}" --parallel "${J}"

echo "=== smoke ==="
"${BUILD}/katago" version
echo "binary: ${BUILD}/katago"
