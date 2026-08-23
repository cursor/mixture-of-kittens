#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/ops/empty_like.h>

using namespace kittens;

namespace utils {

namespace fwd_epilogue {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 256;
    static constexpr int NUM_WARPS = NUM_THREADS / WARP_THREADS;
};

struct globals {
    static constexpr int Nb = 1024; // TMA ignores elements outside of box, so this supports arbitrary shapes
    static constexpr int TOKENS_PER_CTA = 2;

    using token_vec = sv_bf<Nb>;
    using activation_gl = gl<bf16, 1, 1, -1, -1, token_vec>;
    using weight_gl = gl<float, 1, 1, -1, -1>;

    activation_gl y_shared;        // (num_local_tokens, H)
    activation_gl combine_buffer;  // (num_local_tokens * topk, H)
    weight_gl topk_weights;        // (num_local_tokens, topk)
    activation_gl output;          // (num_local_tokens, H)

    __host__ inline dim3 grid() const {
        const int col_blocks = (y_shared.cols() + Nb - 1) / Nb;
        const int token_blocks = y_shared.rows() / TOKENS_PER_CTA;
        return dim3(col_blocks * token_blocks);
    }
    __host__ inline int dynamic_shared_memory() const {
        return TOKENS_PER_CTA * ((topk_weights.cols() + 1) * sizeof(token_vec) + topk_weights.cols() * sizeof(float)) + 1024;
    }
};

static __device__ __forceinline__ void fwd_epilogue_kernel(const globals &g) {
    constexpr int TOKENS_PER_CTA = globals::TOKENS_PER_CTA;
    using compute_group = group<config::NUM_WARPS>;

    const int tid = threadIdx.x;
    const int topk = g.topk_weights.cols();
    const int num_tokens_per_stage = topk + 1;
    const int col_blocks = (g.y_shared.cols() + globals::Nb - 1) / globals::Nb;
    const int col_block_idx = blockIdx.x % col_blocks;
    const int first_token_idx = blockIdx.x / col_blocks * TOKENS_PER_CTA;

    extern __shared__ int __shm[];
    auto *token_vecs = reinterpret_cast<globals::token_vec*>((reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023));
    float *weights = reinterpret_cast<float*>(token_vecs + TOKENS_PER_CTA * num_tokens_per_stage); // (TOKENS_PER_CTA, topk)

    __shared__ semaphore inputs_arrived[TOKENS_PER_CTA];
    if (tid == 0) {
        #pragma unroll
        for (int stage = 0; stage < TOKENS_PER_CTA; ++stage) {
            init_semaphore(inputs_arrived[stage], 0, 1);
            tma::expect_bytes(inputs_arrived[stage], num_tokens_per_stage * sizeof(globals::token_vec));
        }
    }
    for (int i = tid; i < TOKENS_PER_CTA * topk; i += blockDim.x)
        weights[i] = g.topk_weights[{first_token_idx + i / topk, i % topk}];
    __syncthreads();

    #pragma unroll
    for (int stage = 0; stage < TOKENS_PER_CTA; ++stage) {
        const int token_idx = first_token_idx + stage;
        if (tid == 0)
            tma::load_async(token_vecs[stage * num_tokens_per_stage], g.y_shared, {token_idx, col_block_idx}, inputs_arrived[stage]);
        else if (tid < num_tokens_per_stage)
            tma::load_async(token_vecs[stage * num_tokens_per_stage + tid], g.combine_buffer, {token_idx * topk + tid - 1, col_block_idx}, inputs_arrived[stage]);
    }

    #pragma unroll
    for (int stage = 0; stage < TOKENS_PER_CTA; ++stage) {
        globals::token_vec *stage_vecs = token_vecs + stage * num_tokens_per_stage;
        rv_fl<globals::Nb / config::NUM_WARPS> accumulator, term;
        wait(inputs_arrived[stage], 0);
        compute_group::load(accumulator, stage_vecs[0]);
        for (int k = 0; k < topk; ++k) {
            compute_group::load(term, stage_vecs[1 + k]);
            compute_group::mul(term, term, weights[stage * topk + k]);
            compute_group::add(accumulator, accumulator, term);
        }
        compute_group::store(stage_vecs[0], accumulator);
        __syncthreads();
        if (tid == 0)
            tma::store_async(g.output, stage_vecs[0], {first_token_idx + stage, col_block_idx});
    }
}

static __host__ at::Tensor fwd_epilogue_entrypoint(
    const at::Tensor &y_shared,
    const at::Tensor &combine_buffer,
    const at::Tensor &topk_weights
) {
    at::Tensor output = at::empty_like(y_shared);
    globals g {
        .y_shared = kittens::py::tensor_to_gl<globals::activation_gl>(y_shared),
        .combine_buffer = kittens::py::tensor_to_gl<globals::activation_gl>(combine_buffer),
        .topk_weights = kittens::py::tensor_to_gl<globals::weight_gl>(topk_weights),
        .output = kittens::py::tensor_to_gl<globals::activation_gl>(output)
    };
    kittens::py::launch_kernel<config, globals, fwd_epilogue_kernel>(g);
    return output;
}

} // namespace fwd_epilogue

namespace bwd_epilogue {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 256;
    static constexpr int NUM_WARPS = NUM_THREADS / WARP_THREADS;
};

struct globals {
    static constexpr int Nb = 1024; // TMA ignores elements outside of box, so this supports arbitrary shapes

    using token_vec = sv_bf<Nb>;
    using activation_gl = gl<bf16, 1, 1, -1, -1, token_vec>;

    activation_gl d_x_shared;        // (num_local_tokens, H)
    activation_gl d_x_routed_buffer; // (num_local_tokens * topk, H)
    activation_gl d_x;               // (num_local_tokens, H)

    __host__ inline dim3 grid() const {
        const int col_blocks = (d_x_shared.cols() + Nb - 1) / Nb;
        return dim3(col_blocks * d_x_shared.rows());
    }
    __host__ inline int dynamic_shared_memory() const { return (d_x_routed_buffer.rows() / d_x_shared.rows() + 1) * sizeof(token_vec) + 1024; }
};

static __device__ __forceinline__ void bwd_epilogue_kernel(const globals &g) {
    using compute_group = group<config::NUM_WARPS>;

    const int tid = threadIdx.x;
    const int topk = g.d_x_routed_buffer.rows() / g.d_x_shared.rows();
    const int num_vecs = topk + 1;
    const int col_blocks = (g.d_x_shared.cols() + globals::Nb - 1) / globals::Nb;
    const int token_idx = blockIdx.x / col_blocks;
    const int col_block_idx = blockIdx.x % col_blocks;

    extern __shared__ int __shm[];
    auto *token_vecs = reinterpret_cast<globals::token_vec*>((reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023));

    __shared__ semaphore inputs_arrived;
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect_bytes(inputs_arrived, num_vecs * sizeof(globals::token_vec));
    }
    __syncthreads();

    if (tid == 0)
        tma::load_async(token_vecs[0], g.d_x_shared, {token_idx, col_block_idx}, inputs_arrived);
    else if (tid < num_vecs)
        tma::load_async(token_vecs[tid], g.d_x_routed_buffer, {token_idx * topk + tid - 1, col_block_idx}, inputs_arrived);

    rv_fl<globals::Nb / config::NUM_WARPS> accumulator, term;
    wait(inputs_arrived, 0);
    compute_group::load(accumulator, token_vecs[0]);
    for (int k = 0; k < topk; ++k) {
        compute_group::load(term, token_vecs[1 + k]);
        compute_group::add(accumulator, accumulator, term);
    }
    compute_group::store(token_vecs[0], accumulator);
    __syncthreads();
    if (tid == 0)
        tma::store_async(g.d_x, token_vecs[0], {token_idx, col_block_idx});
}

static __host__ at::Tensor bwd_epilogue_entrypoint(const at::Tensor &d_x_shared, const at::Tensor &d_x_routed_buffer) {
    at::Tensor d_x = at::empty_like(d_x_shared);
    globals g {
        .d_x_shared = kittens::py::tensor_to_gl<globals::activation_gl>(d_x_shared),
        .d_x_routed_buffer = kittens::py::tensor_to_gl<globals::activation_gl>(d_x_routed_buffer),
        .d_x = kittens::py::tensor_to_gl<globals::activation_gl>(d_x)
    };
    kittens::py::launch_kernel<config, globals, bwd_epilogue_kernel>(g);
    return d_x;
}

} // namespace bwd_epilogue

namespace all_gather_top_experts {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 1;
};

struct globals {
    int *local_ptr;
    int *multicast_ptr;
    int rank;
    int numel;
    int chunk_bytes;

    __host__ inline dim3 grid() const {
        return dim3(numel * sizeof(int) / chunk_bytes);
    }

    __host__ inline int dynamic_shared_memory() const {
        return chunk_bytes + 1024;
    }
};

__device__ __forceinline__ void all_gather_top_experts_kernel(const globals &G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    int *shared = &al.allocate<int>();
    __shared__ semaphore arrived;
    const int offset = blockIdx.x * G.chunk_bytes / sizeof(int);

    init_semaphore(arrived, 0, 1);
    tma::expect_bytes(arrived, G.chunk_bytes);
    tma::load_async(shared, G.local_ptr + offset, G.chunk_bytes, arrived);
    wait(arrived, 0);
    tma::store_async(G.multicast_ptr + G.rank * G.numel + offset, shared, G.chunk_bytes);
    tma::store_async_wait();
}

static __host__ void all_gather_top_experts_entrypoint(
    const at::Tensor &top_experts,
    const at::Tensor &all_gather_top_experts_buffer,
    int64_t all_gather_top_experts_buffer_multicast_ptr,
    int rank,
    int chunk_bytes
) {
    (void)all_gather_top_experts_buffer;
    const int numel = top_experts.numel();

    globals G {
        .local_ptr = top_experts.data_ptr<int>(),
        .multicast_ptr = reinterpret_cast<int *>(all_gather_top_experts_buffer_multicast_ptr),
        .rank = rank,
        .numel = numel,
        .chunk_bytes = chunk_bytes,
    };
    kittens::py::launch_kernel<config, globals, all_gather_top_experts_kernel>(G);
}

} // namespace all_gather_top_experts

namespace barrier_all {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    uint32_t *multicast_ptr;
    uint32_t *local_ptr;
    uint32_t *target_ptr;
    uint32_t ep_size;
};

__device__ __forceinline__ void barrier_all_kernel(const globals &G) {
    const uint32_t target = atomicAdd(G.target_ptr, G.ep_size) + G.ep_size;

    asm volatile("{multimem.red.release.sys.global.add.u32 [%0], 1;}" :: "l"(G.multicast_ptr) : "memory");
    // The counter is updated through its multicast alias and polled through its unicast alias
    asm volatile("{fence.proxy.alias;}" ::: "memory");

    uint32_t value;
    do {
        asm volatile("{ld.relaxed.sys.global.u32 %0, [%1];}" : "=r"(value) : "l"(G.local_ptr) : "memory");
    } while (value < target);
    asm volatile("{fence.acquire.sys;}" ::: "memory");
}

static __host__ void barrier_all_entrypoint(
    const at::Tensor &barrier_buffer,
    const std::vector<int64_t> &barrier_buffer_ptrs,
    int64_t barrier_buffer_multicast_ptr,
    const at::Tensor &target
) {
    const int64_t barrier_buffer_local_ptr = reinterpret_cast<int64_t>(barrier_buffer.data_ptr<int>());
    const int64_t target_ptr = reinterpret_cast<int64_t>(target.data_ptr<int>());

    globals G {
        .multicast_ptr = reinterpret_cast<uint32_t *>(barrier_buffer_multicast_ptr),
        .local_ptr = reinterpret_cast<uint32_t *>(barrier_buffer_local_ptr),
        .target_ptr = reinterpret_cast<uint32_t *>(target_ptr),
        .ep_size = static_cast<uint32_t>(barrier_buffer_ptrs.size()),
    };
    kittens::py::launch_kernel<config, globals, barrier_all_kernel>(G);
}

} // namespace barrier_all

namespace zero_empty_routed_wgrads {

struct globals {
    uint16_t *d_w_routed_gate;
    uint16_t *d_w_routed_up;
    uint16_t *d_w_routed_down;
    const int *tokens_per_expert;
    const int64_t elements_per_expert;
};

static __global__ void zero_empty_routed_wgrads_kernel(const __grid_constant__ globals g) {
    const int expert_idx = blockIdx.y;
    if (g.tokens_per_expert[expert_idx] != 0) return;
    const int64_t expert_offset = expert_idx * g.elements_per_expert;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < g.elements_per_expert; idx += gridDim.x * blockDim.x) {
        g.d_w_routed_gate[expert_offset + idx] = 0;
        g.d_w_routed_up[expert_offset + idx] = 0;
        g.d_w_routed_down[expert_offset + idx] = 0;
    }
}

} // namespace zero_empty_routed_wgrads

} // namespace utils
