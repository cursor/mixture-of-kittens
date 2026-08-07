#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>

using namespace kittens;

namespace scheduler {

struct config {
    static constexpr int EXPERT_PADDING = 256; // row alignment for contiguous grouped GEMM
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 1024;
    static constexpr int NUM_WARPS = NUM_THREADS / WARP_THREADS;
    static constexpr int ROUTE_BLOCK_THREADS = 256;
    static constexpr int ROUTE_BLOCKS_PER_PEER = NUM_THREADS / ROUTE_BLOCK_THREADS;
};

struct globals {
    using topk_gl = gl<int, 1, -1, -1, -1>;
    using index_gl = gl<int, 1, 1, 1, -1>;

    topk_gl topk;                        // (world_size, num_local_tokens, topk)
    index_gl schedule_peer_rank;         // (schedule_capacity,) must be initialized to -1
    index_gl schedule_peer_token_idx;    // (schedule_capacity,) original_token_idx * topk + k
    index_gl num_tokens;                 // (1,) total padded token count
    index_gl tokens_per_expert;          // (num_local_experts,) padded per-expert token counts
    index_gl tokens_per_expert_and_peer; // (num_local_experts * world_size,) per-(local_expert, peer_rank) token counts
    index_gl expert_offsets;             // (num_local_experts,) base of each expert's padded segment
    index_gl thread_offsets;             // (num_local_experts * world_size * NUM_THREADS,) zero-initialized per-scheduler-thread offsets

    int rank;                            // this (destination) rank
};

// Stage 1: Count routes by local expert, peer rank, and scheduler thread. Every
// counter has one writer, so the result is deterministic without atomics.
static __device__ __forceinline__ void count_kernel(const globals &G) {
    const int world_size = G.topk.depth();
    const int num_local_tokens = G.topk.rows();
    const int topk = G.topk.cols();
    const int rank_stride = num_local_tokens * topk;
    const int num_local_experts = G.tokens_per_expert.cols();
    const int first_expert = G.rank * num_local_experts;
    const int last_expert = first_expert + num_local_experts;
    const int block_idx = static_cast<int>(blockIdx.x);
    const int peer_rank = block_idx / config::ROUTE_BLOCKS_PER_PEER;
    const int thread_group = block_idx % config::ROUTE_BLOCKS_PER_PEER;
    const int tid = thread_group * config::ROUTE_BLOCK_THREADS + threadIdx.x;

    for (int peer_token_idx = tid; peer_token_idx < rank_stride; peer_token_idx += config::NUM_THREADS) {
        const int expert_idx = G.topk.raw_ptr[peer_rank * rank_stride + peer_token_idx];
        if (expert_idx >= first_expert && expert_idx < last_expert) {
            const int idx = ((expert_idx - first_expert) * world_size + peer_rank) * config::NUM_THREADS + tid;
            ++G.thread_offsets.raw_ptr[idx];
        }
    }
}

// Stage 2: Turn the per-thread counts into exclusive offsets. These offsets
// reproduce the old thread-major route order without rescanning for each expert.
static __device__ __forceinline__ void prefix_kernel(const globals &G) {
    const int tid = threadIdx.x;
    const int expert_peer_idx = static_cast<int>(blockIdx.x);
    const int idx = expert_peer_idx * config::NUM_THREADS + tid;
    const int count = G.thread_offsets[{idx}];
    int inclusive = count;

    // Step 2.1: Compute inclusive route counts within each warp.
    #pragma unroll
    for (int offset = 1; offset < WARP_THREADS; offset *= 2) {
        const int n = __shfl_up_sync(0xffffffff, inclusive, offset);
        if (warp::laneid() >= offset) inclusive += n;
    }

    __shared__ int warp_totals[config::NUM_WARPS];
    if (warp::laneid() == WARP_THREADS - 1) warp_totals[warpid()] = inclusive;
    __syncthreads();

    // Step 2.2: Compute inclusive totals across warps.
    if (warpid() == 0) {
        int warp_total = warp::laneid() < config::NUM_WARPS ? warp_totals[warp::laneid()] : 0;
        #pragma unroll
        for (int offset = 1; offset < WARP_THREADS; offset *= 2) {
            const int n = __shfl_up_sync(0xffffffff, warp_total, offset);
            if (warp::laneid() >= offset) warp_total += n;
        }
        if (warp::laneid() < config::NUM_WARPS) warp_totals[warp::laneid()] = warp_total;
    }
    __syncthreads();

    // Step 2.3: Store each thread's exclusive offset and the expert/peer total.
    const int thread_offset = inclusive - count + (warpid() == 0 ? 0 : warp_totals[warpid() - 1]);
    G.thread_offsets[{idx}] = thread_offset;
    if (tid == config::NUM_THREADS - 1)
        G.tokens_per_expert_and_peer[{expert_peer_idx}] = thread_offset + count;
}

// Stage 3: Pad expert segments and compute their offsets.
static __device__ __forceinline__ void finalize_kernel(const globals &G) {
    const int world_size = G.topk.depth();
    const int num_local_experts = G.tokens_per_expert.cols();
    int expert_offset = 0;
    // Preserve expert order while accumulating padded segment bases.
    for (int local_expert = 0; local_expert < num_local_experts; ++local_expert) {
        int expert_tokens = 0;
        for (int peer_rank = 0; peer_rank < world_size; ++peer_rank)
            expert_tokens += G.tokens_per_expert_and_peer[{local_expert * world_size + peer_rank}];
        const int padded_tokens = (expert_tokens + config::EXPERT_PADDING - 1) / config::EXPERT_PADDING * config::EXPERT_PADDING;
        G.expert_offsets[{local_expert}] = expert_offset;
        G.tokens_per_expert[{local_expert}] = padded_tokens;
        expert_offset += padded_tokens;
    }
    G.num_tokens[{0}] = expert_offset;
}

// Stage 4: Visit every route once and place local routes into their padded expert segments.
static __device__ __forceinline__ void schedule_kernel(const globals &G) {
    const int world_size = G.topk.depth();
    const int num_local_tokens = G.topk.rows();
    const int topk = G.topk.cols();
    const int rank_stride = num_local_tokens * topk;
    const int num_local_experts = G.tokens_per_expert.cols();
    const int first_expert = G.rank * num_local_experts;
    const int last_expert = first_expert + num_local_experts;
    const int block_idx = static_cast<int>(blockIdx.x);
    const int peer_rank = block_idx / config::ROUTE_BLOCKS_PER_PEER;
    const int thread_group = block_idx % config::ROUTE_BLOCKS_PER_PEER;
    const int tid = thread_group * config::ROUTE_BLOCK_THREADS + threadIdx.x;

    if (G.num_tokens[{0}] > G.schedule_peer_rank.cols()) asm volatile("{trap;}");

    extern __shared__ int metadata[];
    int *tokens_per_expert_and_peer = metadata;
    int *expert_offsets = metadata + num_local_experts * world_size;
    // Cache the Stage 2 counts and Stage 3 expert bases once per route block.
    for (int idx = threadIdx.x; idx < num_local_experts * world_size; idx += config::ROUTE_BLOCK_THREADS)
        tokens_per_expert_and_peer[idx] = G.tokens_per_expert_and_peer[{idx}];
    for (int idx = threadIdx.x; idx < num_local_experts; idx += config::ROUTE_BLOCK_THREADS)
        expert_offsets[idx] = G.expert_offsets[{idx}];
    __syncthreads();

    for (int peer_token_idx = tid; peer_token_idx < rank_stride; peer_token_idx += config::NUM_THREADS) {
        const int expert_idx = G.topk.raw_ptr[peer_rank * rank_stride + peer_token_idx];
        if (expert_idx >= first_expert && expert_idx < last_expert) {
            const int local_expert = expert_idx - first_expert;
            const int offset_idx = (local_expert * world_size + peer_rank) * config::NUM_THREADS + tid;
            const int j = G.thread_offsets.raw_ptr[offset_idx]++;
            int dst_token_idx = expert_offsets[local_expert];
            // Insert this peer's j-th route after the complete round-robin layers [0, j).
            for (int rank = 0; rank < world_size; ++rank) {
                const int peer_tokens = tokens_per_expert_and_peer[local_expert * world_size + rank];
                dst_token_idx += min(peer_tokens, j);
                dst_token_idx += (rank < peer_rank && peer_tokens > j) ? 1 : 0;
            }
            G.schedule_peer_rank[{dst_token_idx}] = peer_rank;
            G.schedule_peer_token_idx[{dst_token_idx}] = peer_token_idx; // original_token_idx * topk + k
        }
    }
}

static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> schedule(
    const at::Tensor &topk_all,
    const int num_local_experts,
    const int schedule_capacity,
    const int rank
) {
    const int world_size = static_cast<int>(topk_all.size(0));
    const int64_t num_thread_offsets = static_cast<int64_t>(num_local_experts) * world_size * config::NUM_THREADS;

    at::Tensor schedule_peer_rank = at::empty({schedule_capacity}, topk_all.options().dtype(at::kInt));
    at::Tensor schedule_peer_token_idx = at::empty({schedule_capacity}, topk_all.options().dtype(at::kInt));
    at::Tensor num_tokens = at::empty({1}, topk_all.options().dtype(at::kInt));
    at::Tensor tokens_per_expert = at::empty({num_local_experts}, topk_all.options().dtype(at::kInt));
    at::Tensor tokens_per_expert_and_peer = at::empty({num_local_experts * world_size}, topk_all.options().dtype(at::kInt));
    at::Tensor expert_offsets = at::empty({num_local_experts}, topk_all.options().dtype(at::kInt));
    at::Tensor thread_offsets = at::zeros({num_thread_offsets}, topk_all.options().dtype(at::kInt));
    schedule_peer_rank.fill_(-1);

    globals G {
        .topk = kittens::py::tensor_to_gl<globals::topk_gl>(topk_all),
        .schedule_peer_rank = kittens::py::tensor_to_gl<globals::index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<globals::index_gl>(schedule_peer_token_idx),
        .num_tokens = kittens::py::tensor_to_gl<globals::index_gl>(num_tokens),
        .tokens_per_expert = kittens::py::tensor_to_gl<globals::index_gl>(tokens_per_expert),
        .tokens_per_expert_and_peer = kittens::py::tensor_to_gl<globals::index_gl>(tokens_per_expert_and_peer),
        .expert_offsets = kittens::py::tensor_to_gl<globals::index_gl>(expert_offsets),
        .thread_offsets = kittens::py::tensor_to_gl<globals::index_gl>(thread_offsets),
        .rank = rank,
    };

    auto stream = at::cuda::getCurrentCUDAStream();
    kittens::py::global_kernel<config, globals, scheduler::count_kernel>
        <<<world_size * config::ROUTE_BLOCKS_PER_PEER, config::ROUTE_BLOCK_THREADS, 0, stream>>>(G);
    kittens::py::global_kernel<config, globals, scheduler::prefix_kernel>
        <<<num_local_experts * world_size, config::NUM_THREADS, 0, stream>>>(G);
    kittens::py::global_kernel<config, globals, scheduler::finalize_kernel>
        <<<1, 1, 0, stream>>>(G);
    kittens::py::global_kernel<config, globals, scheduler::schedule_kernel>
        <<<world_size * config::ROUTE_BLOCKS_PER_PEER, config::ROUTE_BLOCK_THREADS,
           num_local_experts * (world_size + 1) * sizeof(int), stream>>>(G);

    return {schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert};
}

} // namespace scheduler
