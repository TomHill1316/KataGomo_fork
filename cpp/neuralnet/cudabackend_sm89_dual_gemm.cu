/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Fixed-channel/board wrapper around CUTLASS examples/45_dual_gemm.
 * CUTLASS commit: 7127592069c2fe01b041e174ba4345ef9b279671
 **************************************************************************************************/

#include "../neuralnet/cudabackend_sm89_dual_gemm.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "device/dual_gemm.h"
#include "thread/left_silu_and_mul.h"

namespace Sm89Backend {
namespace {

constexpr int S = 361;
// PATCH(b15-shape): was `constexpr int Channels = 384` (b11c768 mid channels).
// b15c1024 runs the dual FFN at 512 mid channels.
constexpr int Channels = 512;
// PATCH(prune-width): was `constexpr int FfnChannels = 1152`. Variable-width FFN
// models need the width at runtime. Only the problem shape and the leading
// dimensions move to runtime; the CUTLASS tile shapes and the AlignmentA/B = 8
// constraint stay compile-time, so the width must remain a multiple of 8.
constexpr int FfnChannelAlign = 8;

using Element = cutlass::half_t;
using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
  Element, 8, Element, float, cutlass::epilogue::thread::ScaleType::Nothing>;
using SwiGLUOutputOp = cutlass::epilogue::thread::LeftSiLUAndMul<
  Element, 8, Element, float>;

template<int Count>
class Half2TanhSwiGLUOutputOp {
 public:
  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;
  using FragmentOutput = cutlass::Array<Element, Count>;
  using FragmentAccumulator = cutlass::Array<Element, Count>;
  struct Params {};

  CUTLASS_HOST_DEVICE
  explicit Half2TanhSwiGLUOutputOp(Params const&) {}

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return true; }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int, int) { assert(false); }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const& lhs,
    FragmentAccumulator const& rhs
  ) const {
    cutlass::multiplies<FragmentOutput> mul;
    cutlass::multiply_add<FragmentOutput> fma;
    cutlass::fast_tanh_op<FragmentOutput> tanh;
    Element half = cutlass::constants::half<Element>();
    FragmentOutput sigmoid = fma(tanh(mul(lhs, half)), half, half);
    return mul(mul(lhs, sigmoid), rhs);
  }

  CUTLASS_HOST_DEVICE
  Element operator()(Element const& lhs, Element const& rhs) const {
    Element half = cutlass::constants::half<Element>();
    Element sigmoid = cutlass::fast_tanh(lhs * half) * half + half;
    return lhs * sigmoid * rhs;
  }
};

using SwiGLUHalf2TanhOutputOp = Half2TanhSwiGLUOutputOp<8>;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

template<typename SwiGLUOp, int Swizzle>
using DualGemmT = cutlass::gemm::device::DualGemm<
  Element,
  cutlass::layout::RowMajor,
  Element,
  cutlass::layout::RowMajor,
  cutlass::layout::RowMajor,
  Element,
  cutlass::layout::RowMajor,
  Element,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  EpilogueOutputOp,
  SwiGLUOp,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>,
  3,
  false,
  false,
  false,
  8,
  8>;

using DualGemmSwizzle2 = DualGemmT<SwiGLUOutputOp,2>;
using DualGemmSwizzle4 = DualGemmT<SwiGLUOutputOp,4>;
using DualGemmHalf2Tanh = DualGemmT<SwiGLUHalf2TanhOutputOp,2>;

template<typename Gemm>
typename Gemm::Arguments makeArguments(
  const half* weights,
  const half* input,
  half* output,
  int tokens,
  int ffnChannels
) {
  using Layout = cutlass::layout::RowMajor;
  typename Gemm::TensorRefC nullC;
  typename Gemm::TensorRefD nullD;
  return {
    cutlass::gemm::DualGemmMode::kGemm,
    {tokens, ffnChannels, Channels},
    {reinterpret_cast<const Element*>(input), Layout(Channels)},
    {reinterpret_cast<const Element*>(weights), Layout(ffnChannels)},
    nullC,
    nullD,
    {reinterpret_cast<const Element*>(weights + (size_t)ffnChannels * Channels), Layout(ffnChannels)},
    nullC,
    nullD,
    {reinterpret_cast<Element*>(output), Layout(ffnChannels)},
    {1.0f, 0.0f},
    {1.0f, 0.0f},
    {},
    1
  };
}

} // namespace

struct Sm89DualGemmSwiGLU::Impl {
  const half* weights;
  const int ffnChannels;
  DualGemmSwizzle2 swizzle2Op;
  DualGemmSwizzle4 swizzle4Op;
  DualGemmHalf2Tanh half2TanhOp;
  const std::string tactic;
  bool initialized;

  Impl(const half* weights_, const std::string& tactic_, int ffnChannels_)
    : weights(weights_), ffnChannels(ffnChannels_), swizzle2Op(), swizzle4Op(),
      half2TanhOp(), tactic(tactic_), initialized(false)
  {}

  template<typename Gemm>
  bool applyImpl(Gemm& gemm, const half* input, half* output, int tokens, cudaStream_t stream) {
    typename Gemm::Arguments args = makeArguments<Gemm>(weights, input, output, tokens, ffnChannels);
    cutlass::Status status;
    if(!initialized) {
      status = gemm.can_implement(args);
      if(status != cutlass::Status::kSuccess)
        return false;
      status = gemm.initialize(args, nullptr, stream);
      if(status != cutlass::Status::kSuccess)
        return false;
      initialized = true;
    }
    else {
      status = gemm.update(args, nullptr);
      if(status != cutlass::Status::kSuccess)
        return false;
    }
    return gemm.run(stream) == cutlass::Status::kSuccess;
  }

  bool apply(const half* input, half* output, int tokens, cudaStream_t stream) {
    if(tactic == "m128-n64-k32-w64-n32-s3-sw2-exp")
      return applyImpl(swizzle2Op, input, output, tokens, stream);
    if(tactic == "m128-n64-k32-w64-n32-s3-sw4-exp")
      return applyImpl(swizzle4Op, input, output, tokens, stream);
    if(tactic == "m128-n64-k32-w64-n32-s3-sw2-tanh-half2")
      return applyImpl(half2TanhOp, input, output, tokens, stream);
    return false;
  }
};

Sm89DualGemmSwiGLU::Sm89DualGemmSwiGLU(
  const half* weights,
  const std::string& tactic,
  int ffnChannels
)
  : impl(std::make_unique<Impl>(weights, tactic, ffnChannels))
{}

Sm89DualGemmSwiGLU::~Sm89DualGemmSwiGLU() = default;

bool Sm89DualGemmSwiGLU::apply(
  const half* input,
  half* output,
  int batchSize,
  int seqLen,
  int inChannels,
  int ffnChannels,
  cudaStream_t stream
) {
  if(batchSize < 1 || seqLen != S || inChannels != Channels ||
     ffnChannels != impl->ffnChannels || ffnChannels % FfnChannelAlign != 0 ||
     input == nullptr || output == nullptr)
    return false;
  return impl->apply(input, output, batchSize * seqLen, stream);
}

} // namespace Sm89Backend
