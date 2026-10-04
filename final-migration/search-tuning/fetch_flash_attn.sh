#!/usr/bin/env bash
# 取回 SM89 AOT 内核所需的 flash-attention 源码。
#
# 为什么必须取：
#   cpp/CMakeLists.txt:53   set(SM89_FLASH_ATTN_ROOT "" CACHE PATH ...)
#   cpp/CMakeLists.txt:216  if(SM89_FLASH_ATTN_ROOT)  -> 才加入 katago_sm89_flash OBJECT 库
#   cpp/CMakeLists.txt:306-312 由该库置 KATAGO_ENABLE_SM89_{FLASH_ATTN,DUAL_GEMM,LINEAR2_GEMM,
#                              OUTPROJ_GEMM,PRECONV_GEMM,POSTCONV_GEMM,QKV_ROPE_GEMM}=TRUE
#   cpp/CMakeLists.txt:672-688 再把对应宏转发给主 target katago
# 若没有这些宏，cudabackend_sm89_forward.cpp 里的 usedPreConvGemm 等恒为 false，
# 而生产 plan 明确要求
#   cudaPreConvCutlassTacticSm89=m128-n128-k32-w64-n64-s3-sw1
#   cudaPostConvCutlassTacticSm89=m128-n128-k32-w64-n64-s3-sw1
#   cudaLinear2CutlassTacticSm89=m128-n128-k32-w64-n64-s4-sw1
#   cudaDualFfnCutlassTacticSm89=m128-n64-k32-w64-n32-s3-sw2-tanh-half2
#   cudaUseQKVRoPEGemmSm89=true / cudaUsePostConvBNSiluSm89=true
# ⇒ fail-closed 抛 "Selected SM89 preConv CUTLASS tactic is unavailable"。
#
# 版本钉死依据（CMakeLists:225-260 会硬校验，不符即 FATAL_ERROR）：
#   flash-attention 69e1bcbe77c359c84b3a4589e92a7c076e33a202
#   csrc/cutlass   7127592069c2fe01b041e174ba4345ef9b279671
# 注意 cpp/neuralnet/FLASH_ATTENTION_SM89.md 里写的 5835c733… 是旧值，以 CMakeLists 为准。
set -Eeuo pipefail

ROOT="${1:-/root/autodl-tmp/katago-plan/katago-src}"
FA_COMMIT="${FA_COMMIT:-69e1bcbe77c359c84b3a4589e92a7c076e33a202}"
CUTLASS_COMMIT="${CUTLASS_COMMIT:-7127592069c2fe01b041e174ba4345ef9b279671}"
FA_ROOT="${ROOT}/third_party/flash-attention"
CUTLASS_ROOT="${ROOT}/third_party/cutlass"

# AutoDL 直连 GitHub 不通；env 不跨 ssh 会话，所以必须在本进程内 source。
if [ -f /etc/network_turbo ]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi

[ -e "${CUTLASS_ROOT}/include/cutlass/cutlass.h" ] || {
  echo "ERROR: 先跑 tools/fetch_third_party.sh（缺 ${CUTLASS_ROOT}）" >&2; exit 1; }
have_cutlass="$(git -C "${CUTLASS_ROOT}" rev-parse HEAD 2>/dev/null || true)"
[ "${have_cutlass}" = "${CUTLASS_COMMIT}" ] || {
  echo "ERROR: third_party/cutlass HEAD=${have_cutlass} != ${CUTLASS_COMMIT}" >&2; exit 1; }

# ---- 1. flash-attention 本体 -------------------------------------------------
if [ ! -f "${FA_ROOT}/hopper/flash_fwd_launch_template.h" ]; then
  echo "== fetch flash-attention @ ${FA_COMMIT} =="
  rm -rf "${FA_ROOT}"
  mkdir -p "${FA_ROOT}"
  git -C "${FA_ROOT}" init -q
  git -C "${FA_ROOT}" remote add origin https://github.com/Dao-AILab/flash-attention.git
  git -C "${FA_ROOT}" fetch -q --depth 1 origin "${FA_COMMIT}"
  git -C "${FA_ROOT}" checkout -q FETCH_HEAD
else
  echo "== flash-attention 已存在，跳过 fetch =="
fi

# CMake 优先读 .katago-source-revision（CMakeLists:229-238），写它可绕过 git rev-parse，
# 也让以后重新 checkout 不会误判。
printf '%s\n' "${FA_COMMIT}" > "${FA_ROOT}/.katago-source-revision"

# ---- 2. csrc/cutlass --------------------------------------------------------
# 不取子模块：third_party/cutlass 已是同一个 commit（git 内容寻址 ⇒ 树完全一致）。
# 用相对符号链接，零拷贝；CMake 的 EXISTS 检查会穿透符号链接。
if [ ! -e "${FA_ROOT}/csrc/cutlass/include/cutlass/cutlass.h" ]; then
  echo "== 链接 csrc/cutlass -> ../../cutlass =="
  mkdir -p "${FA_ROOT}/csrc"
  rm -rf "${FA_ROOT}/csrc/cutlass"
  ln -sfn ../../cutlass "${FA_ROOT}/csrc/cutlass"
fi
# 符号链接下 CMake 走 git rev-parse 分支；显式写 marker 更稳（优先级更高）。
printf '%s\n' "${CUTLASS_COMMIT}" > "${FA_ROOT}/csrc/cutlass/.katago-source-revision"

# ---- 3. 打 SM89 patch -------------------------------------------------------
# 改的是 hopper/{tile_size.h,flash_fwd_launch_template.h,mainloop_fwd_sm80.hpp,softmax.h}。
if grep -q "KATAGO_FLASH_BLOCK_N" "${FA_ROOT}/hopper/tile_size.h" 2>/dev/null &&
   grep -q "KATAGO_FLASH_BOTH16_ACCUM" "${FA_ROOT}/hopper/flash_fwd_launch_template.h" 2>/dev/null; then
  echo "== patch 已应用，跳过 =="
else
  echo "== 应用 cpp/neuralnet/flash-attention-sm89.patch =="
  if git -C "${FA_ROOT}" apply --check "${ROOT}/cpp/neuralnet/flash-attention-sm89.patch" 2>/dev/null; then
    git -C "${FA_ROOT}" apply "${ROOT}/cpp/neuralnet/flash-attention-sm89.patch"
  else
    echo "   git apply --check 不通过，改用 patch -p1 --forward"
    patch -p1 --forward --directory="${FA_ROOT}" \
      < "${ROOT}/cpp/neuralnet/flash-attention-sm89.patch"
  fi
fi

# ---- 4. 复刻 CMake 的全部硬检查 ---------------------------------------------
echo "== 校验（与 CMakeLists:217-271 一致） =="
fail=0
chk() { if [ -e "$2" ]; then echo "  OK   $1"; else echo "  FAIL $1  ($2)"; fail=1; fi; }
chk "hopper/flash_fwd_launch_template.h"        "${FA_ROOT}/hopper/flash_fwd_launch_template.h"
chk "hopper/tile_size.h"                        "${FA_ROOT}/hopper/tile_size.h"
chk "hopper/mainloop_fwd_sm80.hpp"              "${FA_ROOT}/hopper/mainloop_fwd_sm80.hpp"
chk "hopper/softmax.h"                          "${FA_ROOT}/hopper/softmax.h"
chk "LICENSE"                                   "${FA_ROOT}/LICENSE"
chk "csrc/cutlass/include/cutlass/cutlass.h"    "${FA_ROOT}/csrc/cutlass/include/cutlass/cutlass.h"
chk "csrc/cutlass/examples/45_dual_gemm/…"      "${FA_ROOT}/csrc/cutlass/examples/45_dual_gemm/device/dual_gemm.h"

for marker in \
  "hopper/tile_size.h:KATAGO_FLASH_BLOCK_N" \
  "hopper/flash_fwd_launch_template.h:KATAGO_FLASH_BOTH16_ACCUM" \
  "hopper/mainloop_fwd_sm80.hpp:KATAGO_FLASH_BOTH16_ACCUM" \
  "hopper/softmax.h:KATAGO_FLASH_BOTH16_ACCUM"; do
  f="${FA_ROOT}/${marker%%:*}"; m="${marker##*:}"
  if grep -q "$m" "$f"; then echo "  OK   patch marker $m in ${marker%%:*}";
  else echo "  FAIL patch marker $m 缺于 ${marker%%:*}"; fail=1; fi
done

echo "  FA  .katago-source-revision = $(cat "${FA_ROOT}/.katago-source-revision")"
echo "  CUT .katago-source-revision = $(cat "${FA_ROOT}/csrc/cutlass/.katago-source-revision")"
echo "  git rev-parse csrc/cutlass   = $(git -C "${FA_ROOT}/csrc/cutlass" rev-parse HEAD 2>/dev/null || echo '(n/a)')"
du -sh "${FA_ROOT}" 2>/dev/null

[ "$fail" = 0 ] || { echo "== 校验失败 ==" >&2; exit 1; }
echo "== done: SM89_FLASH_ATTN_ROOT=${FA_ROOT} =="
