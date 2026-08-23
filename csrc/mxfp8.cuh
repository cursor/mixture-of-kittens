#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/ops/empty.h>

using namespace kittens;

namespace mxfp8 {

struct config {
    static constexpr int Mb = 128;
    static constexpr int Nb = 128;

    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = Mb;
};

struct globals {
    using x_bf16_tile = st_bf<config::Mb, config::Nb, false>;
    using x_fp8_tile = st_fp8e4m3<config::Mb, config::Nb, false>;
    using x_sc_tile = st_fp8e8m0<32, 16, false>;

    using x_bf16_gl = gl<bf16, 1, -1, -1, -1, x_bf16_tile>;
    using x_fp8_gl = gl<fp8e4m3, 1, -1, -1, -1, x_fp8_tile>;
    using x_sc_gl = gl<fp8e8m0, -1, -1, 32, 16, x_sc_tile>;

    x_bf16_gl x_bf16;      // (E, M, N) bf16 input (E = 1 for 2D inputs)
    x_fp8_gl x_fp8;        // (E, M, N) MXFP8 output
    x_sc_gl x_sc;          // (E * M // 128, N // 128, 32, 16) fp8e8m0 block scales
    x_fp8_gl x_fp8_t;      // (E, N, M) MXFP8 output, transposed
    x_sc_gl x_sc_t;        // (E * N // 128, M // 128, 32, 16) fp8e8m0 block scales, transposed

    __host__ inline dim3 grid() const {
        return dim3(x_bf16.cols() / config::Nb, x_bf16.rows() / config::Mb, x_bf16.depth());
    }
    __host__ inline int dynamic_shared_memory() const {
        return config::Mb * config::Nb * sizeof(bf16)
             + config::Mb * config::Nb * sizeof(fp8e4m3) + 32 * 16 * sizeof(fp8e8m0)
             + 1024;
    }
};

static __device__ __forceinline__ void quantize_single_block(
    const bf16_2 (&values_bf16)[16],
    uint32_t (&values_fp8)[8],
    uint32_t &scale_byte
) {
    bf16_2 amax = __habs2(values_bf16[0]);
    #pragma unroll
    for (int i = 1; i < 16; i++)
        amax = __hmax2(amax, __habs2(values_bf16[i]));

    // Compute the e8m0 scale, rounding towards positive infinity and saturating to finite (https://arxiv.org/pdf/2506.08027)
    const float scale = max(__bfloat162float(__hmax(amax.x, amax.y)) * 0.002232142857f, 0.000000000001f);
    uint16_t scale_fp8x2;
    asm volatile("{cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;}" : "=h"(scale_fp8x2) : "f"(scale), "f"(scale));
    scale_byte = scale_fp8x2 & 0xFF;
    const float scale_inv = __uint_as_float((254u - scale_byte) << 23); // directly build float32 reciprocal without division

    // Quantize
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const float2 v01_fp32 = __bfloat1622float2(values_bf16[i * 2]);
        const float2 v23_fp32 = __bfloat1622float2(values_bf16[i * 2 + 1]);
        uint16_t v01_fp8, v23_fp8;
        asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}" : "=h"(v01_fp8) : "f"(v01_fp32.x * scale_inv), "f"(v01_fp32.y * scale_inv));
        asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}" : "=h"(v23_fp8) : "f"(v23_fp32.x * scale_inv), "f"(v23_fp32.y * scale_inv));
        values_fp8[i] = static_cast<uint32_t>(v01_fp8) | (static_cast<uint32_t>(v23_fp8) << 16);
    }
}

static __device__ __forceinline__ void dequantize_single_block(
    const uint32_t (&values_fp8)[8],
    const uint32_t scale_byte,
    float2 (&values_fp32)[16]
) {
    const float scale = __uint_as_float(scale_byte << 23);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint16_t v01_fp8 = static_cast<uint16_t>(values_fp8[i] >> (h * 16));
            const __half2_raw v01_fp16 = __nv_cvt_fp8x2_to_halfraw2(v01_fp8, __NV_E4M3); // no direct conversion from fp8 -> fp32
            const float2 v01_fp32 = __half22float2(*reinterpret_cast<const __half2 *>(&v01_fp16));
            values_fp32[i * 2 + h] = float2{v01_fp32.x * scale, v01_fp32.y * scale};
        }
    }
}

template <bool RETURN_NORMAL, bool RETURN_TRANSPOSED, int SRC_ROW_STRIDE = 128, bool SPLIT_PASSES = false, bool SCALE_ROWS = false>
static __device__ __forceinline__ void quantize_tile(
    const globals::x_bf16_tile &x_bf16_tile,
    const globals::x_fp8_tile &x_fp8_tile,
    const globals::x_sc_tile &x_sc_tile,
    const globals::x_fp8_tile &x_fp8_t_tile,
    const globals::x_sc_tile &x_sc_t_tile,
    const float *row_scales,
    const int _tid,
    const int barrier_id
) {
    constexpr int TILE_SIZE = 128;
    constexpr int K_BLOCK_SIZE = 32;
    static_assert(RETURN_NORMAL || RETURN_TRANSPOSED, "At least one output pair must be requested");
    static_assert(!SPLIT_PASSES || (RETURN_NORMAL && RETURN_TRANSPOSED), "SPLIT_PASSES requires both outputs");

    // Excess threads have no tile rows to handle. Caller must ensure that this is called by threads 0-127, or 0-255 with SPLIT_PASSES
    const int tid = SPLIT_PASSES ? _tid % TILE_SIZE : _tid;
    if (!SPLIT_PASSES && tid >= TILE_SIZE) return;

    constexpr int NUM_K_BLOCKS = TILE_SIZE / K_BLOCK_SIZE; // 4
    constexpr int PACKED_PER_K_BLOCK = K_BLOCK_SIZE / 2;   // 16

    const uint32_t bf16_src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&x_bf16_tile));

    #pragma unroll
    for (int pass = 0; pass < 2; pass++) {
        const bool transposed = pass == 0;
        if ((transposed && !RETURN_TRANSPOSED) || (!transposed && !RETURN_NORMAL)) continue;
        if (SPLIT_PASSES && transposed != (_tid < TILE_SIZE)) continue;
        const uint32_t fp8_dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(transposed ? &x_fp8_t_tile : &x_fp8_tile));
        const uint32_t sc_dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(transposed ? &x_sc_t_tile : &x_sc_tile));
        const int row = transposed ? (tid % 64) * 2 + tid / 64 : tid;

        auto load_k_block = [&](int k_block_idx, bf16_2 (&dst)[PACKED_PER_K_BLOCK]) {
            // Load one 32-element block of this thread's row from shared memory w/ custom swizzling
            #pragma unroll
            for (int k = 0; k < PACKED_PER_K_BLOCK; k++) {
                const int col = k_block_idx*K_BLOCK_SIZE + (tid*4 + k*2) % K_BLOCK_SIZE;
                if (transposed) {
                    move<bf16>::lds(dst[k].x, bf16_src_addr + (col*SRC_ROW_STRIDE + row) * sizeof(bf16));
                    move<bf16>::lds(dst[k].y, bf16_src_addr + ((col+1)*SRC_ROW_STRIDE + row) * sizeof(bf16));
                    if constexpr (SCALE_ROWS) {
                        const float2 val = __bfloat1622float2(dst[k]);
                        dst[k] = __float22bfloat162_rn(float2{val.x * row_scales[col], val.y * row_scales[col + 1]});
                    }
                } else {
                    move<bf16_2>::lds(dst[k], bf16_src_addr + (row*SRC_ROW_STRIDE + col) * sizeof(bf16));
                    if constexpr (SCALE_ROWS) {
                        const float2 val = __bfloat1622float2(dst[k]);
                        dst[k] = __float22bfloat162_rn(float2{val.x * row_scales[row], val.y * row_scales[row]});
                    }
                }
            }
        };
        auto quantize_k_block = [&](int k_block_idx, const bf16_2 (&src)[PACKED_PER_K_BLOCK], uint32_t &scale_word) {
            // Quantize one 32-element block and store the FP8 output to shared memory
            uint32_t x_fp8_reg[PACKED_PER_K_BLOCK / 2];
            uint32_t scale_byte;
            quantize_single_block(src, x_fp8_reg, scale_byte);
            scale_word |= scale_byte << (k_block_idx * 8);
            #pragma unroll
            for (int k = 0; k < PACKED_PER_K_BLOCK / 2; k++) {
                const int col = k_block_idx*K_BLOCK_SIZE + (tid*4 + k*4) % K_BLOCK_SIZE;
                move<int>::sts(fp8_dst_addr + row*TILE_SIZE + col, std::bit_cast<int>(x_fp8_reg[k]));
            }
        };

        uint32_t scale_word = 0;
        if (!transposed) {
            // This writes FP8 output in place over the BF16 source, so it must load the whole tile up front
            bf16_2 x_bf16_reg[NUM_K_BLOCKS][PACKED_PER_K_BLOCK];
            #pragma unroll
            for (int j = 0; j < NUM_K_BLOCKS; j++)
                load_k_block((j + tid/8) % NUM_K_BLOCKS, x_bf16_reg[j]);
            group<TILE_SIZE / WARP_THREADS>::sync(barrier_id); // in-place writes may begin
            #pragma unroll
            for (int j = 0; j < NUM_K_BLOCKS; j++)
                quantize_k_block((j + tid/8) % NUM_K_BLOCKS, x_bf16_reg[j], scale_word);
        } else {
            #pragma unroll 1 // otherwise we get hundreds of register spills
            for (int j = 0; j < NUM_K_BLOCKS; j++) {
                const int k_block_idx = (j + tid/8) % NUM_K_BLOCKS;
                bf16_2 x_bf16_reg[PACKED_PER_K_BLOCK];
                load_k_block(k_block_idx, x_bf16_reg);
                quantize_k_block(k_block_idx, x_bf16_reg, scale_word);
            }
        }

        // Store the scales to shared memory. Each thread will access 1 bank, so no need to swizzle,
        // but we do have to follow this complicated layout pattern made by NVIDIA:
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
        move<int>::sts(sc_dst_addr + (row % 32) * 16 + (row / 32) * 4, std::bit_cast<int>(scale_word));
    }
}

template <bool RETURN_NORMAL, bool RETURN_TRANSPOSED>
static __device__ __forceinline__ void quantize_kernel(const globals &G) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    const uint64_t smem_base_addr = (reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023);
    auto &x_bf16_tile = *reinterpret_cast<globals::x_bf16_tile *>(smem_base_addr);
    auto &x_fp8_tile = *reinterpret_cast<globals::x_fp8_tile *>(&x_bf16_tile);
    auto &x_sc_tile = *reinterpret_cast<globals::x_sc_tile *>(reinterpret_cast<uint64_t>(&x_bf16_tile) + sizeof(globals::x_fp8_tile));
    auto &x_fp8_t_tile = *reinterpret_cast<globals::x_fp8_tile *>(reinterpret_cast<uint64_t>(&x_bf16_tile) + sizeof(globals::x_bf16_tile));
    auto &x_sc_t_tile = *reinterpret_cast<globals::x_sc_tile *>(reinterpret_cast<uint64_t>(&x_fp8_t_tile) + sizeof(globals::x_fp8_tile));

    // Calculate indices
    const int tid = threadIdx.x;
    const int expert = blockIdx.z;
    const int row = blockIdx.y;
    const int col = blockIdx.x;

    // Initialize mbarrier and initiate TMA load
    __shared__ semaphore inputs_arrived;
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect(inputs_arrived, x_bf16_tile);
        tma::load_async(x_bf16_tile, G.x_bf16, {expert, row, col}, inputs_arrived);
    }

    // Wait for the TMA load to complete
    __syncthreads();
    wait(inputs_arrived, 0);

    // Quantize
    quantize_tile<RETURN_NORMAL, RETURN_TRANSPOSED>(x_bf16_tile, x_fp8_tile, x_sc_tile, x_fp8_t_tile, x_sc_t_tile, nullptr, tid, 1);
    __syncthreads();

    // Store to global memory
    if (tid == 0) {
        if constexpr (RETURN_TRANSPOSED) {
            tma::store_async(G.x_fp8_t, x_fp8_t_tile, {expert, col, row});
            tma::store_async(G.x_sc_t, x_sc_t_tile, {expert * static_cast<int>(gridDim.x) + col, row, 0, 0});
        }
        if constexpr (RETURN_NORMAL) {
            tma::store_async(G.x_fp8, x_fp8_tile, {expert, row, col});
            tma::store_async(G.x_sc, x_sc_tile, {expert * static_cast<int>(gridDim.y) + row, col, 0, 0});
        }
    }
}

static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
quantize_entrypoint(
    const at::Tensor &x_bf16,
    const bool return_normal,
    const bool return_transposed
) {
    using C = mxfp8::config;
    using G = mxfp8::globals;

    const int E = x_bf16.dim() == 3 ? static_cast<int>(x_bf16.size(0)) : 1;
    const int M = static_cast<int>(x_bf16.size(-2));
    const int N = static_cast<int>(x_bf16.size(-1));

    at::Tensor x_fp8, x_sc, x_fp8_t, x_sc_t;
    if (return_normal) {
        x_fp8 = at::empty(x_bf16.sizes(), x_bf16.options().dtype(at::kFloat8_e4m3fn));
        x_sc = at::empty({E * (M / 128), N / 128, 32, 16}, x_bf16.options().dtype(at::kByte));
    }
    if (return_transposed) {
        x_fp8_t = x_bf16.dim() == 3 ? at::empty({E, N, M}, x_bf16.options().dtype(at::kFloat8_e4m3fn))
                                    : at::empty({N, M}, x_bf16.options().dtype(at::kFloat8_e4m3fn));
        x_sc_t = at::empty({E * (N / 128), M / 128, 32, 16}, x_bf16.options().dtype(at::kByte));
    }

    G g {
        .x_bf16 = kittens::py::tensor_to_gl<G::x_bf16_gl>(x_bf16),
        .x_fp8 = return_normal ? kittens::py::tensor_to_gl<G::x_fp8_gl>(x_fp8)
                               : kittens::py::make_fake_gl<G::x_fp8_gl>(1, E, M, N),
        .x_sc = return_normal ? kittens::py::tensor_to_gl<G::x_sc_gl>(x_sc)
                              : kittens::py::make_fake_gl<G::x_sc_gl>(E * (M / 128), N / 128, 32, 16),
        .x_fp8_t = return_transposed ? kittens::py::tensor_to_gl<G::x_fp8_gl>(x_fp8_t)
                                     : kittens::py::make_fake_gl<G::x_fp8_gl>(1, E, N, M),
        .x_sc_t = return_transposed ? kittens::py::tensor_to_gl<G::x_sc_gl>(x_sc_t)
                                    : kittens::py::make_fake_gl<G::x_sc_gl>(E * (N / 128), M / 128, 32, 16)
    };

    if (return_normal && return_transposed)
        kittens::py::launch_kernel<C, G, mxfp8::quantize_kernel<true, true>>(g);
    else if (return_normal)
        kittens::py::launch_kernel<C, G, mxfp8::quantize_kernel<true, false>>(g);
    else
        kittens::py::launch_kernel<C, G, mxfp8::quantize_kernel<false, true>>(g);

    return {x_fp8, x_sc, x_fp8_t, x_sc_t};
}

} // namespace mxfp8
