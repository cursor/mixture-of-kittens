#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include "../mxfp8.cuh"
#include "../utils.cuh"
#include "situglu.cuh"

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>

using namespace kittens;

enum class RoutedPrecision {
    BF16,
    MXFP8,
};

template <int NUM_DEVICES,
          RoutedPrecision ROUTED_PRECISION = RoutedPrecision::MXFP8,
          GluActivation ACTIVATION = GluActivation::SWIGLU>
struct dispatch_mlp_swiglu_combiner {

static constexpr bool USE_MXFP8 = ROUTED_PRECISION == RoutedPrecision::MXFP8;
static constexpr bool USE_SITUGLU = ACTIVATION == GluActivation::SITUGLU;

#include "types.cuh"

#include "utils.cuh"
#include "dispatch_combine.cuh"
#include "swiglu.cuh"
#include "grouped_gemm.cuh"

#include "forward.cuh"
#include "backward.cuh"
#include "recompute_forward_context.cuh"

}; // struct dispatch_mlp_swiglu_combiner
