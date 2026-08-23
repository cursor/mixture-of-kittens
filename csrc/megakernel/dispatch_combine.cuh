template <bool SCALE_ROWS = false>
static __device__ __forceinline__ void dispatch_kernel(
    const activation_bf16_pgl &peer_buf,
    const routed_activation_gl &x_gmem,
    const routed_sc_gl *x_sc_gmem,
    const routed_transposed_gl *x_t_gmem,
    const routed_sc_gl *x_sc_t_gmem,
    const router_weight_gl *router_weights,
    const index_gl &schedule_peer_rank,
    const index_gl &schedule_peer_token_idx,
    const index_gl *transfer_ready,
    const index_gl *buffer_ready,
    const index_gl &transfer_done,
    semaphore &inputs_arrived,
    uint32_t &bitfield,
    const int num_tokens,
    const int macrobatch_size,
    const int minibatch_size,
    const int macrobatch_idx,
    const int task_idx,
    const int row_divisor,
    const int previous_macrobatch_idx,
    const int buffer_ready_required_count,
    const uint64_t smem_base_addr
) {
    auto &token_chunks        = *reinterpret_cast<bf16 (*)[config::DISPATCH_Mb][config::DISPATCH_Nb]>(smem_base_addr);
    auto &x_fp8_tiles         = *reinterpret_cast<quant_fp8_tile (*)[config::DISPATCH_OUT_TILES]>(smem_base_addr + sizeof(token_chunks));
    auto &x_sc_tiles          = *reinterpret_cast<quant_sc_tile (*)[config::DISPATCH_OUT_TILES]>(smem_base_addr + sizeof(token_chunks) + sizeof(x_fp8_tiles));
    auto &x_fp8_t_tiles       = *reinterpret_cast<quant_fp8_tile (*)[config::DISPATCH_OUT_TILES]>(smem_base_addr + sizeof(token_chunks) + sizeof(x_fp8_tiles) + sizeof(x_sc_tiles));
    auto &x_sc_t_tiles        = *reinterpret_cast<quant_sc_tile (*)[config::DISPATCH_OUT_TILES]>(smem_base_addr + sizeof(token_chunks) + sizeof(x_fp8_tiles) + sizeof(x_sc_tiles) + sizeof(x_fp8_t_tiles));
    auto &router_weights_smem = *reinterpret_cast<float (*)[config::DISPATCH_Mb]>(smem_base_addr + sizeof(token_chunks) + sizeof(x_fp8_tiles) + sizeof(x_sc_tiles) + sizeof(x_fp8_t_tiles) + sizeof(x_sc_t_tiles));

    const int tid = threadIdx.x;
    const bool is_worker = tid < config::DISPATCH_Mb; // only these threads move tokens, but all threads join the barriers and waits

    const int cols = x_gmem.cols();
    const int col_blocks = (cols + config::DISPATCH_Nb - 1) / config::DISPATCH_Nb;

    const int macrobatch_offset = macrobatch_idx * macrobatch_size;
    const int num_macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_offset);
    if (task_idx >= num_macrobatch_tokens / config::DISPATCH_Mb * col_blocks) return;

    const int row_idx = task_idx / col_blocks * config::DISPATCH_Mb;
    const int col_block_idx = task_idx % col_blocks;
    const int chunk_cols = min(config::DISPATCH_Nb, cols - col_block_idx * config::DISPATCH_Nb);
    const uint32_t chunk_bytes = chunk_cols * sizeof(bf16);

    const int peer_rank = is_worker ? schedule_peer_rank[{macrobatch_offset + row_idx + tid}] : -1;
    const int peer_token_idx = is_worker ? schedule_peer_token_idx[{macrobatch_offset + row_idx + tid}] : -1;
    const int num_valid = __syncthreads_count(peer_rank >= 0);

    if (tid == 0) {
        if (transfer_ready != nullptr) {
            // Dispatch can overwrite rows still read by the previously processed macrobatch's GEMMs.
            const int previous_macrobatch_offset = previous_macrobatch_idx * macrobatch_size;
            const int previous_macrobatch_tokens = min(macrobatch_size, num_tokens - previous_macrobatch_offset);
            if (row_idx < previous_macrobatch_tokens) { // otherwise, previous macrobatch was partial & no need to wait
                const int global_minibatch_idx = (previous_macrobatch_offset + row_idx) / minibatch_size;
                const int minibatch_rows = min(minibatch_size, num_tokens - global_minibatch_idx * minibatch_size);
                const int required_count = ((minibatch_rows + config::MLP_Mb - 1) / config::MLP_Mb) * (cols / config::MLP_Nb) * config::CLUSTER_SIZE;
                barrier_wait(*transfer_ready, global_minibatch_idx, required_count);
            }
        }
        if (buffer_ready != nullptr)
            barrier_wait(*buffer_ready, previous_macrobatch_idx, buffer_ready_required_count);
        tma::expect_bytes(inputs_arrived, num_valid * chunk_bytes);
    }
    __syncthreads();

    if constexpr (SCALE_ROWS) {
        if (is_worker) router_weights_smem[tid] = peer_rank >= 0 ? router_weights->raw_ptr[row_idx + tid] : 0.0f;
    }

    if (peer_rank >= 0) {
        tma::load_async(token_chunks[tid],
                        &peer_buf[peer_rank][static_cast<size_t>(peer_token_idx / row_divisor) * cols + col_block_idx * config::DISPATCH_Nb],
                        chunk_bytes, inputs_arrived);
    } else if (is_worker) { // zero-fill padding rows for correct transpose-quantize
        auto *chunk = reinterpret_cast<float4 *>(token_chunks[tid]);
        #pragma unroll
        for (int i = 0; i < config::DISPATCH_Nb * static_cast<int>(sizeof(bf16)) / static_cast<int>(sizeof(float4)); ++i)
            chunk[i] = float4{0.0f, 0.0f, 0.0f, 0.0f};
    }

    wait(inputs_arrived, get_phasebit<0>(bitfield, 0));
    update_phasebit<0>(bitfield, 0);

    if constexpr (USE_MXFP8) {
        // Quantize each 128x128 subtile of the staging buffer
        const int row_block = row_idx / config::QUANT_Mb;
        const int num_subtiles = chunk_cols / config::QUANT_Nb;
        for (int subtile = 0; subtile < config::DISPATCH_Nb / config::QUANT_Nb; ++subtile) {
            if (subtile < num_subtiles) {
                const auto &x_bf16_subtile = *reinterpret_cast<const quant_bf16_tile *>(
                    smem_base_addr + subtile * config::QUANT_Nb * sizeof(bf16));
                const int out = subtile % config::DISPATCH_OUT_TILES;

                if (tid == 0) tma::store_async_read_wait<4 * (config::DISPATCH_OUT_TILES - 1)>();
                __syncthreads(); // also makes zero-filled rows visible before the first transpose-quantize
                if constexpr (!SCALE_ROWS)
                    mxfp8::quantize_tile<true, true, config::DISPATCH_Nb, true, false>(x_bf16_subtile, x_fp8_tiles[out], x_sc_tiles[out], x_fp8_t_tiles[out], x_sc_t_tiles[out], nullptr, tid, 1);
                else
                    mxfp8::quantize_tile<true, true, config::DISPATCH_Nb, true, true> (x_bf16_subtile, x_fp8_tiles[out], x_sc_tiles[out], x_fp8_t_tiles[out], x_sc_t_tiles[out], router_weights_smem, tid, 1);
                __syncthreads(); // quantized tiles must be complete before TMA reads them

                if (tid == 0) {
                    const int col_block = col_block_idx * (config::DISPATCH_Nb / config::QUANT_Nb) + subtile;
                    tma::store_async(x_gmem, x_fp8_tiles[out], {row_block, col_block});
                    tma::store_async(*x_sc_gmem, x_sc_tiles[out], {row_block, col_block, 0, 0});
                    tma::store_async(*x_t_gmem, x_fp8_t_tiles[out], {col_block, row_block});
                    tma::store_async(*x_sc_t_gmem, x_sc_t_tiles[out], {col_block, row_block, 0, 0});
                }
            }
        }
    } else {
        if constexpr (SCALE_ROWS) {
            if (is_worker && peer_rank >= 0) {
                auto *pairs = reinterpret_cast<bf16_2 *>(token_chunks[tid]);
                const float weight = router_weights_smem[tid];
                for (int col = 0; col < chunk_cols / 2; ++col) {
                    float2 value = __bfloat1622float2(pairs[col]);
                    pairs[col] = __floats2bfloat162_rn(value.x * weight, value.y * weight);
                }
            }
            __syncthreads();
        }
        if (is_worker)
            tma::store_async(&x_gmem.raw_ptr[static_cast<size_t>(row_idx + tid) * cols + col_block_idx * config::DISPATCH_Nb],
                             token_chunks[tid], chunk_bytes);
    }

    // Bulk groups are per-thread: every BF16 worker issues a store, while only
    // thread 0 issues the MXFP8 stores.
    if constexpr (USE_MXFP8) {
        if (tid == 0) {
            tma::store_async_wait();
            const int global_minibatch_idx = (macrobatch_offset + row_idx) / minibatch_size;
            barrier_arrive(transfer_done, global_minibatch_idx);
        }
        __syncthreads(); // the next task on this CTA reuses the staging buffer
    } else {
        if (is_worker)
            tma::store_async_wait();
        __syncthreads(); // stores are visible and the staging buffer is reusable
        if (tid == 0) {
            const int global_minibatch_idx = (macrobatch_offset + row_idx) / minibatch_size;
            barrier_arrive(transfer_done, global_minibatch_idx);
        }
    }
}

template <bool ARRIVE_BY_ROW = false>
static __device__ __forceinline__ void combine_kernel(
    const activation_bf16_pgl &peer_buf,
    const epi_bf16_gl &local_buf,
    const router_weight_pgl *d_router_weight_buffer,
    const d_router_weight_partials_gl *d_router_weight_partials,
    const index_gl &schedule_peer_rank,
    const index_gl &schedule_peer_token_idx,
    const index_gl &transfer_ready,
    const index_gl *transfer_done,
    semaphore (&inputs_arrived)[config::COMBINE_PIPE_DEPTH],
    uint32_t &bitfield,
    const int num_tokens,
    const int macrobatch_size,
    const int minibatch_size,
    const int macrobatch_idx,
    const int task_idx,
    const uint64_t smem_base_addr
) {
    auto &token_chunks = *reinterpret_cast<bf16 (*)[config::COMBINE_PIPE_DEPTH][config::COMBINE_Mb][config::COMBINE_Nb]>(smem_base_addr);

    const int tid = threadIdx.x;
    const bool is_worker = tid < config::COMBINE_Mb; // only these threads move tokens, but all threads join the barriers and waits

    const int cols = local_buf.cols();
    const int col_blocks = (cols + config::COMBINE_Nb - 1) / config::COMBINE_Nb;
    const int first_tile_idx = task_idx * config::COMBINE_PIPE_DEPTH;

    const int macrobatch_offset = macrobatch_idx * macrobatch_size;
    const int num_macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_offset);
    const int num_valid_tiles = min(config::COMBINE_PIPE_DEPTH, num_macrobatch_tokens / config::COMBINE_Mb * col_blocks - first_tile_idx); // because we pad to 256
    if (num_valid_tiles <= 0) return;

    const int first_row_idx = first_tile_idx / col_blocks * config::COMBINE_Mb + tid;
    const int first_col_block_idx = first_tile_idx % col_blocks;

    int row_idx[config::COMBINE_PIPE_DEPTH], col_block_idx[config::COMBINE_PIPE_DEPTH], peer_rank[config::COMBINE_PIPE_DEPTH], 
        peer_token_idx[config::COMBINE_PIPE_DEPTH], num_valid[config::COMBINE_PIPE_DEPTH];
    #pragma unroll
    for (int stage = 0, row = first_row_idx, col = first_col_block_idx; stage < config::COMBINE_PIPE_DEPTH; ++stage) {
        const bool is_valid_tile = stage < num_valid_tiles;
        row_idx[stage] = row;
        col_block_idx[stage] = col;
        peer_rank[stage] = is_valid_tile && is_worker ? schedule_peer_rank[{macrobatch_offset + row}] : -1;
        peer_token_idx[stage] = is_valid_tile && is_worker ? schedule_peer_token_idx[{macrobatch_offset + row}] : -1;
        num_valid[stage] = !is_valid_tile ? 0
                         : (stage == 0 || col == 0) ? __syncthreads_count(peer_rank[stage] >= 0)
                         : num_valid[stage - 1];
        if (++col == col_blocks) { col = 0; row += config::COMBINE_Mb; }
    }

    auto chunk_bytes = [&](int col_block) {
        return static_cast<uint32_t>(min(config::COMBINE_Nb, cols - col_block * config::COMBINE_Nb) * sizeof(bf16));
    };

    if (tid == 0) {
        // Wait until the GEMMs have fully written every minibatch this task reads
        const int first_global_minibatch_idx = (macrobatch_offset + first_row_idx) / minibatch_size;
        const int last_global_minibatch_idx = (macrobatch_offset + (first_tile_idx + num_valid_tiles - 1) / col_blocks * config::COMBINE_Mb) / minibatch_size;
        for (int global_minibatch_idx = first_global_minibatch_idx; global_minibatch_idx <= last_global_minibatch_idx; ++global_minibatch_idx) {
            const int minibatch_rows = min(minibatch_size, num_tokens - global_minibatch_idx * minibatch_size);
            const int required_count = ((minibatch_rows + config::MLP_Mb - 1) / config::MLP_Mb) * (cols / config::MLP_Nb) * config::CLUSTER_SIZE;
            barrier_wait(transfer_ready, global_minibatch_idx, required_count);
        }
        #pragma unroll
        for (int stage = 0; stage < config::COMBINE_PIPE_DEPTH; ++stage)
            if (stage < num_valid_tiles)
                tma::expect_bytes(inputs_arrived[stage], num_valid[stage] * chunk_bytes(col_block_idx[stage]));
    }
    __syncthreads();

    #pragma unroll
    for (int stage = 0; stage < config::COMBINE_PIPE_DEPTH; ++stage)
        if (peer_rank[stage] >= 0)
            tma::load_async(token_chunks[stage][tid],
                            &local_buf.raw_ptr[static_cast<size_t>(row_idx[stage]) * cols + col_block_idx[stage] * config::COMBINE_Nb],
                            chunk_bytes(col_block_idx[stage]), inputs_arrived[stage]);

    // Store each tile out as its loads arrive
    #pragma unroll
    for (int stage = 0; stage < config::COMBINE_PIPE_DEPTH; ++stage) {
        if (stage < num_valid_tiles) {
            wait(inputs_arrived[stage], get_phasebit<0>(bitfield, stage)); // semaphores are reused across tasks
            update_phasebit<0>(bitfield, stage);
            if (peer_rank[stage] >= 0) {
                tma::store_async(&peer_buf[peer_rank[stage]][static_cast<size_t>(peer_token_idx[stage]) * cols + col_block_idx[stage] * config::COMBINE_Nb],
                                 token_chunks[stage][tid], chunk_bytes(col_block_idx[stage]));
                if (d_router_weight_buffer != nullptr && col_block_idx[stage] == 0) {
                    float d_router_weight = 0.0f;
                    for (int col = 0; col < d_router_weight_partials->cols(); ++col)
                        d_router_weight += (*d_router_weight_partials)[{row_idx[stage], col}];
                    (*d_router_weight_buffer)[peer_rank[stage]][peer_token_idx[stage]] = d_router_weight;
                }
            }
        }
    }

    if constexpr (ARRIVE_BY_ROW) {
        if (transfer_done != nullptr) {
            const int stage = warpid();
            if (warp::laneid() == 0 && stage < num_valid_tiles) {
                const int row = (first_tile_idx + stage) / col_blocks * config::COMBINE_Mb;
                barrier_arrive(*transfer_done, (macrobatch_offset + row) / (config::MLP_Mb / config::CLUSTER_SIZE));
            }
        }
    }

    // The next task on this CTA reuses token_chunks; make sure outgoing stores are done reading shared memory
    tma::store_async_read_wait();
    __syncthreads();
    if constexpr (!ARRIVE_BY_ROW) {
        if (tid == 0 && transfer_done != nullptr)
            barrier_arrive(*transfer_done, macrobatch_idx);
    }
}
