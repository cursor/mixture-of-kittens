static __device__ __forceinline__ void barrier_wait(const index_gl &counter, int index, int required_count) {
    int value;
    while (true) {
        asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(value) : "l"(&counter[{index}]) : "memory");
        if (value >= required_count) break;
        __nanosleep(16);
    }
    asm volatile("{fence.acquire.gpu;}" ::: "memory");
}

static __device__ __forceinline__ void barrier_arrive(const index_gl &counter, int index, int increment = 1) {
    asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&counter[{index}]), "r"(increment) : "memory");
}

static __device__ __forceinline__ void preload_router_weights_kernel(
    const router_weight_pgl &peer_buf,
    const router_weight_gl &router_weights,
    const index_gl &schedule_peer_rank,
    const index_gl &schedule_peer_token_idx,
    const index_gl *buffer_ready,
    const index_gl &transfer_done,
    const int num_tokens,
    const int macrobatch_size,
    const int macrobatch_idx,
    const int task_idx,
    const int num_workers,
    const int previous_macrobatch_idx,
    const int buffer_ready_required_count
) {
    if (threadIdx.x == 0 && buffer_ready != nullptr)
        barrier_wait(*buffer_ready, previous_macrobatch_idx, buffer_ready_required_count);
    __syncthreads();

    const int macrobatch_offset = macrobatch_idx * macrobatch_size;
    const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_offset);
    for (int row = task_idx * config::NUM_THREADS + threadIdx.x; row < macrobatch_tokens; row += num_workers * config::NUM_THREADS) {
        const int peer_rank = schedule_peer_rank[{macrobatch_offset + row}];
        const int peer_token_idx = schedule_peer_token_idx[{macrobatch_offset + row}];
        router_weights.raw_ptr[row] = peer_rank >= 0 ? peer_buf[peer_rank][peer_token_idx] : 0.0f;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        barrier_arrive(transfer_done, macrobatch_idx);
        barrier_wait(transfer_done, macrobatch_idx, num_workers); // all comm SM barrier
    }
    __syncthreads();
}
