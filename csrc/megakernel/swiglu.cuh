template <bool IS_CLAMPED>
static __device__ __forceinline__ float2 swiglu_bwd_pair(
    float2 gate,
    float2 up,
    const float2 d_hidden,
    const float limit,
    bf16_2 &d_gate,
    bf16_2 &d_up
) {
    uint32_t clamp_grad_mask;
    if constexpr (IS_CLAMPED) {
        clamp_grad_mask = static_cast<uint32_t>(gate.x <= limit)
                          | static_cast<uint32_t>(gate.y <= limit) << 1
                          | static_cast<uint32_t>(up.x >= -limit && up.x <= limit) << 2
                          | static_cast<uint32_t>(up.y >= -limit && up.y <= limit) << 3;
        gate = {fminf(gate.x, limit), fminf(gate.y, limit)};
        up = {fminf(fmaxf(up.x, -limit), limit), fminf(fmaxf(up.y, -limit), limit)};
    }
    const float2 sigmoid = {1.0f / (1.0f + __expf(-gate.x)), 1.0f / (1.0f + __expf(-gate.y))};
    const float2 silu = {gate.x * sigmoid.x, gate.y * sigmoid.y};
    const float2 dsilu = {(1.0f - silu.x) * sigmoid.x + silu.x, (1.0f - silu.y) * sigmoid.y + silu.y};
    if constexpr (IS_CLAMPED) {
        d_gate = __floats2bfloat162_rn(
            clamp_grad_mask & 1 ? dsilu.x * up.x * d_hidden.x : 0.0f,
            clamp_grad_mask & 2 ? dsilu.y * up.y * d_hidden.y : 0.0f
        );
        d_up = __floats2bfloat162_rn(
            clamp_grad_mask & 4 ? silu.x * d_hidden.x : 0.0f,
            clamp_grad_mask & 8 ? silu.y * d_hidden.y : 0.0f
        );
    } else {
        d_gate = __floats2bfloat162_rn(dsilu.x * up.x * d_hidden.x, dsilu.y * up.y * d_hidden.y);
        d_up = __floats2bfloat162_rn(silu.x * d_hidden.x, silu.y * d_hidden.y);
    }
    return {silu.x * up.x, silu.y * up.y};
}

template <bool IS_SHARED, bool IS_CLAMPED>
static __device__ __forceinline__ void swiglu_fwd_kernel(
    const epi_bf16_gl &gate_gmem,
    const epi_bf16_gl &up_gmem,
    const std::conditional_t<IS_SHARED, mlp_bf16_gl, routed_activation_gl> &hidden_gmem,
    const routed_sc_gl *hidden_sc_gmem,          // routed MXFP8 only
    const routed_transposed_gl *hidden_t_gmem,   // routed MXFP8 only
    const routed_sc_gl *hidden_sc_t_gmem,        // routed MXFP8 only
    const index_gl &gate_up_tile_ready,
    const index_gl &hidden_row_block_ready,
    semaphore (&swiglu_inputs_arrived)[config::SWIGLU_FWD_PIPE_DEPTH],
    uint32_t &swiglu_bitfield,
    const int num_tokens,
    const float swiglu_limit,
    const int macrobatch_size,
    const int minibatch_size,
    const int macrobatch_idx,
    const int minibatch_idx,
    const int task_idx,
    const int cta_rank,
    const int gate_up_tile_ready_base_index,
    const int hidden_row_block_ready_base_index,
    const uint64_t smem_base_addr
) {
    static constexpr bool USE_ROUTED_MXFP8 = !IS_SHARED && USE_MXFP8;
    using input_bf16_tile = std::conditional_t<USE_ROUTED_MXFP8, quant_bf16_tile, swiglu_tile>;
    using output_bf16_tile = std::conditional_t<USE_ROUTED_MXFP8, quant_bf16_tile, swiglu_tile>;
    auto (&gate_smem)[config::SWIGLU_FWD_PIPE_DEPTH] = *reinterpret_cast<input_bf16_tile (*)[config::SWIGLU_FWD_PIPE_DEPTH]>(smem_base_addr);
    auto (&up_smem)[config::SWIGLU_FWD_PIPE_DEPTH]   = *reinterpret_cast<input_bf16_tile (*)[config::SWIGLU_FWD_PIPE_DEPTH]>(reinterpret_cast<uint64_t>(&gate_smem) + sizeof(gate_smem));
    auto &hidden_smem                                = *reinterpret_cast<output_bf16_tile *>(reinterpret_cast<uint64_t>(&up_smem) + sizeof(up_smem));

    const int intermediate_dim_col_blocks = hidden_gmem.cols() / config::MLP_Nb;
    const int global_minibatch_idx = macrobatch_idx * (macrobatch_size / minibatch_size) + minibatch_idx;
    const int macrobatch_row_block_offset = macrobatch_idx * (macrobatch_size / config::SWIGLU_Mb);

    const int row_blocks = num_tokens / config::SWIGLU_Mb;
    const int col_blocks = hidden_gmem.cols() / config::SWIGLU_Nb;
    const int num_tiles = row_blocks * col_blocks;
    int first_tile_idx, tile_end;
    if constexpr (IS_SHARED) {
        first_tile_idx = task_idx * config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH + cta_rank * config::SWIGLU_FWD_PIPE_DEPTH;
        tile_end = num_tiles;
    } else {
        const int num_tiles_per_minibatch = (minibatch_size / config::SWIGLU_Mb) * col_blocks;
        const int minibatch_first_tile_idx = global_minibatch_idx * num_tiles_per_minibatch;
        first_tile_idx = minibatch_first_tile_idx + task_idx * config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH + cta_rank * config::SWIGLU_FWD_PIPE_DEPTH;
        tile_end = min(num_tiles, minibatch_first_tile_idx + num_tiles_per_minibatch);
    }
    if (first_tile_idx >= tile_end)
        return;

    const int first_row = first_tile_idx / col_blocks;
    const int first_col = first_tile_idx % col_blocks;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < config::SWIGLU_FWD_PIPE_DEPTH; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < tile_end) {
                // This improves throughput
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks) {
                    ++row;
                    col -= col_blocks;
                }
                tma::expect_bytes(swiglu_inputs_arrived[stage], sizeof(gate_smem[stage]) + sizeof(up_smem[stage]));

                const int parent_task_idx = (row / (config::MLP_Mb / config::SWIGLU_Mb)) * intermediate_dim_col_blocks + col / (config::MLP_Nb / config::SWIGLU_Nb);
                barrier_wait(gate_up_tile_ready, gate_up_tile_ready_base_index + parent_task_idx, 2 * config::CLUSTER_SIZE);

                tma::load_async(gate_smem[stage], gate_gmem, {row - macrobatch_row_block_offset, col}, swiglu_inputs_arrived[stage]);
                tma::load_async(up_smem[stage],   up_gmem,   {row - macrobatch_row_block_offset, col}, swiglu_inputs_arrived[stage]);
            }
        }
    }

    using compute_group = group<config::NUM_WARPS>;
    #pragma unroll (IS_SHARED ? config::SWIGLU_FWD_PIPE_DEPTH : 1)
    for (int stage = 0; stage < config::SWIGLU_FWD_PIPE_DEPTH; ++stage) {
        const int tile_idx = first_tile_idx + stage;
        if (tile_idx < tile_end) {
            wait(swiglu_inputs_arrived[stage], get_phasebit<0>(swiglu_bitfield, stage));
            update_phasebit<0>(swiglu_bitfield, stage);

            // This improves throughput
            int row = first_row;
            int col = first_col + stage;
            if (col >= col_blocks) {
                ++row;
                col -= col_blocks;
            }

            if constexpr (!USE_ROUTED_MXFP8) {
                if constexpr (USE_SITUGLU) {
                    if (threadIdx.x == 0) tma::store_async_read_wait();
                    __syncthreads();
                    const auto *gate_pairs = reinterpret_cast<const bf16_2 *>(gate_smem[stage].data);
                    const auto *up_pairs = reinterpret_cast<const bf16_2 *>(up_smem[stage].data);
                    auto *hidden_pairs = reinterpret_cast<bf16_2 *>(hidden_smem.data);
                    #pragma unroll
                    for (int i = threadIdx.x; i < config::SWIGLU_Mb * config::SWIGLU_Nb / 2; i += config::NUM_THREADS) {
                        const float2 gate = __bfloat1622float2(gate_pairs[i]);
                        const float2 up = __bfloat1622float2(up_pairs[i]);
                        hidden_pairs[i] = __floats2bfloat162_rn(
                            situglu_forward(gate.x, up.x),
                            situglu_forward(gate.y, up.y));
                    }
                } else {
                    rt_fl<config::SWIGLU_Mb / config::NUM_WARPS, config::SWIGLU_Nb> gate, up, denominator;
                    compute_group::load(gate, gate_smem[stage]);
                    compute_group::load(up, up_smem[stage]);
                    if constexpr (IS_CLAMPED) {
                        compute_group::min(gate, gate, swiglu_limit);
                        compute_group::max(up, up, -swiglu_limit);
                        compute_group::min(up, up, swiglu_limit);
                    }
                    compute_group::mul(denominator, gate, -1.0f);
                    compute_group::exp(denominator, denominator);
                    compute_group::add(denominator, denominator, 1.0f);
                    compute_group::div(gate, gate, denominator);
                    compute_group::mul(gate, gate, up);
                    if (threadIdx.x == 0) tma::store_async_read_wait();
                    __syncthreads();
                    compute_group::store(hidden_smem, gate);
                }
            } else {
                const auto *gate_pairs = reinterpret_cast<const bf16_2 *>(gate_smem[stage].data);
                const auto *up_pairs = reinterpret_cast<const bf16_2 *>(up_smem[stage].data);
                auto *hidden_pairs = reinterpret_cast<bf16_2 *>(hidden_smem.data);
                #pragma unroll
                for (int i = threadIdx.x; i < config::SWIGLU_Mb * config::SWIGLU_Nb / 2; i += config::NUM_THREADS) {
                    float2 gate = __bfloat1622float2(gate_pairs[i]);
                    float2 up = __bfloat1622float2(up_pairs[i]);
                    if constexpr (USE_SITUGLU) {
                        gate = float2{situglu_forward(gate.x, up.x), situglu_forward(gate.y, up.y)};
                    } else {
                        if constexpr (IS_CLAMPED) {
                            gate = {fminf(gate.x, swiglu_limit), fminf(gate.y, swiglu_limit)};
                            up = {fminf(fmaxf(up.x, -swiglu_limit), swiglu_limit), fminf(fmaxf(up.y, -swiglu_limit), swiglu_limit)};
                        }
                        float2 denominator = base_ops::mul::op<float2>(gate, float2{-1.0f, -1.0f});
                        denominator = base_ops::exp::op<float2>(denominator);
                        denominator = base_ops::sum::op<float2>(denominator, float2{1.0f, 1.0f});
                        gate = base_ops::div::op<float2>(gate, denominator);
                        gate = base_ops::mul::op<float2>(gate, up);
                    }
                    hidden_pairs[i] = __floats2bfloat162_rn(gate.x, gate.y);
                }
            }
            __syncthreads();

            if constexpr (USE_ROUTED_MXFP8) {
                auto &hidden_quant_smem = reinterpret_cast<quant_bf16_tile &>(hidden_smem);
                auto &hidden_fp8_smem = reinterpret_cast<quant_fp8_tile &>(gate_smem[stage]);
                auto &hidden_sc_smem = *reinterpret_cast<quant_sc_tile *>(reinterpret_cast<uint64_t>(&gate_smem[stage]) + sizeof(hidden_fp8_smem));
                auto &hidden_fp8_t_smem = reinterpret_cast<quant_fp8_tile &>(up_smem[stage]);
                auto &hidden_sc_t_smem = *reinterpret_cast<quant_sc_tile *>(reinterpret_cast<uint64_t>(&up_smem[stage]) + sizeof(hidden_fp8_t_smem));

                mxfp8::quantize_tile<true, true>(hidden_quant_smem, hidden_fp8_smem, hidden_sc_smem, hidden_fp8_t_smem, hidden_sc_t_smem, nullptr, threadIdx.x, 1);
                __syncthreads();

                if (threadIdx.x == 0) {
                    tma::store_async(hidden_gmem, hidden_fp8_smem, {row - macrobatch_row_block_offset, col});
                    tma::store_async(*hidden_sc_gmem, hidden_sc_smem, {row - macrobatch_row_block_offset, col, 0, 0});
                    tma::store_async(*hidden_t_gmem, hidden_fp8_t_smem, {col, row - macrobatch_row_block_offset});
                    tma::store_async(*hidden_sc_t_gmem, hidden_sc_t_smem, {col, row - macrobatch_row_block_offset, 0, 0});
                }
            } else if (threadIdx.x == 0) {
                tma::store_async(hidden_gmem, hidden_smem, {row - macrobatch_row_block_offset, col});
            }
        }
    }

    if (threadIdx.x == 0) {
        tma::store_async_wait();
        #pragma unroll
        for (int stage = 0; stage < config::SWIGLU_FWD_PIPE_DEPTH; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < tile_end) {
                // This improves throughput
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks)
                    ++row;
                barrier_arrive(hidden_row_block_ready, hidden_row_block_ready_base_index + row / (config::MLP_Mb / config::SWIGLU_Mb));
            }
        }
    }
}

template <bool IS_SHARED, bool IS_CLAMPED>
static __device__ __forceinline__ void swiglu_bwd_kernel(
    const epi_bf16_gl &d_hidden_gmem,
    const std::conditional_t<IS_SHARED, swiglu_bf16_gl, routed_gate_up_gl> &gate_gmem,
    const std::conditional_t<IS_SHARED, swiglu_bf16_gl, routed_gate_up_gl> &up_gmem,
    const std::conditional_t<IS_SHARED, mlp_bf16_gl, routed_activation_gl> &d_gate_gmem,
    const std::conditional_t<IS_SHARED, mlp_bf16_gl, routed_activation_gl> &d_up_gmem,
    const routed_sc_gl *gate_sc_gmem,            // routed MXFP8 only
    const routed_sc_gl *up_sc_gmem,              // routed MXFP8 only
    const routed_sc_gl *d_gate_sc_gmem,          // routed MXFP8 only
    const routed_sc_gl *d_up_sc_gmem,            // routed MXFP8 only
    const routed_transposed_gl *d_gate_t_gmem,   // routed MXFP8 only
    const routed_sc_gl *d_gate_sc_t_gmem,        // routed MXFP8 only
    const routed_transposed_gl *d_up_t_gmem,     // routed MXFP8 only
    const routed_sc_gl *d_up_sc_t_gmem,          // routed MXFP8 only
    const router_weight_gl *router_weights,
    const d_router_weight_partials_gl *d_router_weight_partials,
    const index_gl *schedule_peer_rank,
    const index_gl &d_hidden_tile_ready,
    const index_gl *replayed_gate_up_tile_ready,
    const index_gl &d_gate_up_row_block_ready,
    const index_gl *buffer_done,
    semaphore (&swiglu_inputs_arrived)[config::SWIGLU_BWD_PIPE_DEPTH],
    uint32_t &swiglu_bitfield,
    const int num_tokens,
    const float swiglu_limit,
    const int macrobatch_size,
    const int minibatch_size,
    const int macrobatch_idx,
    const int minibatch_idx,
    const int task_idx,
    const int cta_rank,
    const int d_hidden_tile_ready_base_index,
    const int replayed_gate_up_tile_ready_base_index,
    const int d_gate_up_row_block_ready_base_index,
    const int buffer_done_index,
    const uint64_t smem_base_addr
) {
    static constexpr bool USE_ROUTED_MXFP8 = !IS_SHARED && USE_MXFP8;
    using gate_up_tile = std::conditional_t<USE_ROUTED_MXFP8, quant_fp8_tile, swiglu_tile>;
    using d_hidden_tile = std::conditional_t<USE_ROUTED_MXFP8, quant_bf16_tile, swiglu_tile>;

    auto &d_hidden_smem  = *reinterpret_cast<d_hidden_tile (*)[config::SWIGLU_BWD_PIPE_DEPTH]>(smem_base_addr);
    auto &gate_smem      = *reinterpret_cast<gate_up_tile (*)[config::SWIGLU_BWD_PIPE_DEPTH]>(reinterpret_cast<uint64_t>(&d_hidden_smem) + sizeof(d_hidden_smem));
    auto &up_smem        = *reinterpret_cast<gate_up_tile (*)[config::SWIGLU_BWD_PIPE_DEPTH]>(reinterpret_cast<uint64_t>(&gate_smem) + sizeof(gate_smem));
    auto &gate_sc_smem   = *reinterpret_cast<quant_sc_tile (*)[config::SWIGLU_BWD_PIPE_DEPTH]>(reinterpret_cast<uint64_t>(&up_smem) + sizeof(up_smem));
    auto &up_sc_smem     = *reinterpret_cast<quant_sc_tile (*)[config::SWIGLU_BWD_PIPE_DEPTH]>(reinterpret_cast<uint64_t>(&gate_sc_smem) + sizeof(gate_sc_smem));
    auto &d_up_bf16_smem = *reinterpret_cast<quant_bf16_tile *>(reinterpret_cast<uint64_t>(&up_sc_smem) + sizeof(up_sc_smem)); // staging shared across stages
    auto &d_t_fp8_smem   = *reinterpret_cast<quant_fp8_tile (*)[2]>(reinterpret_cast<uint64_t>(&d_up_bf16_smem) + sizeof(d_up_bf16_smem));
    auto &d_t_sc_smem    = *reinterpret_cast<quant_sc_tile (*)[2]>(reinterpret_cast<uint64_t>(&d_t_fp8_smem) + sizeof(d_t_fp8_smem));
    const uint64_t router_smem_addr = USE_ROUTED_MXFP8
        ? reinterpret_cast<uint64_t>(&d_t_sc_smem) + sizeof(d_t_sc_smem)
        : reinterpret_cast<uint64_t>(&up_smem) + sizeof(up_smem);
    auto &router_smem    = *reinterpret_cast<float (*)[config::SWIGLU_BWD_PIPE_DEPTH][config::SWIGLU_Mb]>(router_smem_addr);

    const int intermediate_dim_col_blocks = gate_gmem.cols() / config::MLP_Nb;
    const int global_minibatch_idx = macrobatch_idx * (macrobatch_size / minibatch_size) + minibatch_idx;
    const int macrobatch_row_block_offset = macrobatch_idx * (macrobatch_size / config::SWIGLU_Mb);

    const int row_blocks = num_tokens / config::SWIGLU_Mb;
    const int col_blocks = gate_gmem.cols() / config::SWIGLU_Nb;
    const int num_tiles = row_blocks * col_blocks;
    int first_tile_idx, tile_end;
    if constexpr (IS_SHARED) {
        first_tile_idx = task_idx * config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH + cta_rank * config::SWIGLU_BWD_PIPE_DEPTH;
        tile_end = num_tiles;
    } else {
        const int num_tiles_per_minibatch = (minibatch_size / config::SWIGLU_Mb) * col_blocks;
        const int minibatch_first_tile_idx = global_minibatch_idx * num_tiles_per_minibatch;
        first_tile_idx = minibatch_first_tile_idx + task_idx * config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH + cta_rank * config::SWIGLU_BWD_PIPE_DEPTH;
        tile_end = min(num_tiles, minibatch_first_tile_idx + num_tiles_per_minibatch);
    }
    if (first_tile_idx >= tile_end) {
        if (buffer_done != nullptr && threadIdx.x == 0)
            barrier_arrive(*buffer_done, buffer_done_index);
        return;
    }

    const int first_row = first_tile_idx / col_blocks;
    const int first_col = first_tile_idx % col_blocks;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < config::SWIGLU_BWD_PIPE_DEPTH; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < tile_end) {
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks) {
                    ++row;
                    col -= col_blocks;
                }
                if constexpr (USE_ROUTED_MXFP8)
                    tma::expect_bytes(swiglu_inputs_arrived[stage], sizeof(d_hidden_smem[stage]) + sizeof(gate_smem[stage]) + sizeof(up_smem[stage]) + sizeof(gate_sc_smem[stage]) + sizeof(up_sc_smem[stage]));
                else
                    tma::expect_bytes(swiglu_inputs_arrived[stage], sizeof(d_hidden_smem[stage]) + sizeof(gate_smem[stage]) + sizeof(up_smem[stage]));

                const int parent_task_idx = (row / (config::MLP_Mb / config::SWIGLU_Mb)) * intermediate_dim_col_blocks + col / (config::MLP_Nb / config::SWIGLU_Nb);
                barrier_wait(d_hidden_tile_ready, d_hidden_tile_ready_base_index + parent_task_idx, config::CLUSTER_SIZE);
                if (replayed_gate_up_tile_ready != nullptr) // replayed macrobatch: wait for the replayed gate/up GEMMs
                    barrier_wait(*replayed_gate_up_tile_ready, replayed_gate_up_tile_ready_base_index + parent_task_idx, 2 * config::CLUSTER_SIZE);

                tma::load_async(d_hidden_smem[stage], d_hidden_gmem, {row - macrobatch_row_block_offset, col}, swiglu_inputs_arrived[stage]);
                tma::load_async(gate_smem[stage],     gate_gmem,     {row - macrobatch_row_block_offset, col}, swiglu_inputs_arrived[stage]);
                tma::load_async(up_smem[stage],       up_gmem,       {row - macrobatch_row_block_offset, col}, swiglu_inputs_arrived[stage]);
                if constexpr (USE_ROUTED_MXFP8) {
                    tma::load_async(gate_sc_smem[stage], *gate_sc_gmem, {row - macrobatch_row_block_offset, col, 0, 0}, swiglu_inputs_arrived[stage]);
                    tma::load_async(up_sc_smem[stage],   *up_sc_gmem,   {row - macrobatch_row_block_offset, col, 0, 0}, swiglu_inputs_arrived[stage]);
                }
            }
        }
    }

    using compute_group = group<config::NUM_WARPS>;
    #pragma unroll
    for (int stage = 0; stage < config::SWIGLU_BWD_PIPE_DEPTH; ++stage) {
        const int tile_idx = first_tile_idx + stage;
        if (tile_idx < tile_end) {
            wait(swiglu_inputs_arrived[stage], get_phasebit<0>(swiglu_bitfield, stage));
            update_phasebit<0>(swiglu_bitfield, stage);

            int row = first_row;
            int col = first_col + stage;
            if (col >= col_blocks) {
                ++row;
                col -= col_blocks;
            }

            if constexpr (USE_ROUTED_MXFP8) {
                constexpr int MXFP8_Kb = 32;
                const int tile_row = threadIdx.x % config::SWIGLU_Mb; // 2 threads per row
                const int tile_col_half = threadIdx.x / config::SWIGLU_Mb; // first half vs second half

                // For router-weight gradient: each row's two threads read the same weight
                const int global_token_idx = row * config::SWIGLU_Mb + tile_row;
                const int local_token_idx = (row - macrobatch_row_block_offset) * config::SWIGLU_Mb + tile_row;
                const int peer_rank = (*schedule_peer_rank)[{global_token_idx}];
                const float router_weight = router_weights->raw_ptr[local_token_idx];
                const float inv_router_weight = router_weight > 0.0f ? 1.0f / router_weight : 0.0f;
                float router_grad_partial = 0.0f;

                const uint32_t gate_fp8_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&gate_smem[stage]));
                const uint32_t up_fp8_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&up_smem[stage]));
                const uint32_t d_hidden_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&d_hidden_smem[stage]));
                const uint32_t d_up_bf16_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&d_up_bf16_smem));

                int gate_scale_word, up_scale_word;
                const int scale_word_offset = (tile_row % 32) * 16 + (tile_row / 32) * 4;
                move<int>::lds(gate_scale_word, static_cast<uint32_t>(__cvta_generic_to_shared(&gate_sc_smem[stage])) + scale_word_offset);
                move<int>::lds(up_scale_word, static_cast<uint32_t>(__cvta_generic_to_shared(&up_sc_smem[stage])) + scale_word_offset);

                const int k_block_pair = ((tile_row >> 1) & 1) ^ tile_col_half;
                uint32_t d_gate_scale_byte[2], d_up_scale_byte[2];
                #pragma unroll 1
                for (int j = 0; j < 2; ++j) { // 64 elements per thread = 2 32-element blocks per thread
                    const int k_block_idx = k_block_pair * 2 + ((tile_row + j) & 1);

                    // Load this block's gate/up values and the incoming d_hidden gradient
                    uint32_t gate_fp8[8], up_fp8[8];
                    float2 d_hidden_fp32[16];
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const int col_idx = k_block_idx * MXFP8_Kb + ((tile_row / 4 + k) % 8) * 4;
                        move<int>::lds(reinterpret_cast<int &>(gate_fp8[k]), gate_fp8_addr + tile_row * config::SWIGLU_Nb + col_idx);
                        move<int>::lds(reinterpret_cast<int &>(up_fp8[k]), up_fp8_addr + tile_row * config::SWIGLU_Nb + col_idx);
                        float2 d_hidden_packed;
                        move<float2>::lds(d_hidden_packed, d_hidden_addr + (tile_row * config::SWIGLU_Nb + col_idx) * static_cast<int>(sizeof(bf16)));
                        const bf16_2 *d_hidden_pairs = reinterpret_cast<const bf16_2 *>(&d_hidden_packed);
                        d_hidden_fp32[k * 2] = __bfloat1622float2(d_hidden_pairs[0]);
                        d_hidden_fp32[k * 2 + 1] = __bfloat1622float2(d_hidden_pairs[1]);
                    }

                    // Dequantize
                    float2 gate_fp32[16], up_fp32[16];
                    mxfp8::dequantize_single_block(gate_fp8, (static_cast<uint32_t>(gate_scale_word) >> (k_block_idx * 8)) & 0xFF, gate_fp32);
                    mxfp8::dequantize_single_block(up_fp8, (static_cast<uint32_t>(up_scale_word) >> (k_block_idx * 8)) & 0xFF, up_fp32);

                    // Apply the selected GLU derivative in fp32 before requantizing the gradients
                    bf16_2 d_gate_bf16[16], d_up_bf16[16];
                    #pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        const float d_hidden_x = d_hidden_fp32[k].x;
                        const float d_hidden_y = d_hidden_fp32[k].y;
                        if constexpr (USE_SITUGLU) {
                            const situglu_backward_terms terms_x = situglu_backward(gate_fp32[k].x, up_fp32[k].x);
                            const situglu_backward_terms terms_y = situglu_backward(gate_fp32[k].y, up_fp32[k].y);
                            router_grad_partial += d_hidden_x * inv_router_weight * terms_x.hidden
                                                 + d_hidden_y * inv_router_weight * terms_y.hidden;
                            d_gate_bf16[k] = __floats2bfloat162_rn(
                                terms_x.d_gate * d_hidden_x,
                                terms_y.d_gate * d_hidden_y);
                            d_up_bf16[k] = __floats2bfloat162_rn(
                                terms_x.d_up * d_hidden_x,
                                terms_y.d_up * d_hidden_y);
                        } else {
                            const float2 hidden = swiglu_bwd_pair<IS_CLAMPED>(gate_fp32[k], up_fp32[k], d_hidden_fp32[k], swiglu_limit, d_gate_bf16[k], d_up_bf16[k]);
                            router_grad_partial += d_hidden_x * inv_router_weight * hidden.x
                                                 + d_hidden_y * inv_router_weight * hidden.y;
                        }
                    }

                    // Quantize both gradients; also stage them in BF16 for the transpose-quantize below
                    uint32_t d_gate_fp8[8], d_up_fp8[8];
                    mxfp8::quantize_single_block(d_gate_bf16, d_gate_fp8, d_gate_scale_byte[j]);
                    mxfp8::quantize_single_block(d_up_bf16, d_up_fp8, d_up_scale_byte[j]);
                    const uint32_t *d_gate_bf16_words = reinterpret_cast<const uint32_t *>(d_gate_bf16);
                    const uint32_t *d_up_bf16_words = reinterpret_cast<const uint32_t *>(d_up_bf16);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const int col_idx = k_block_idx * MXFP8_Kb + ((tile_row / 4 + k) % 8) * 4;
                        move<int>::sts(gate_fp8_addr + tile_row * config::SWIGLU_Nb + col_idx, std::bit_cast<int>(d_gate_fp8[k]));
                        move<int>::sts(up_fp8_addr + tile_row * config::SWIGLU_Nb + col_idx, std::bit_cast<int>(d_up_fp8[k]));
                        move<float2>::sts(d_hidden_addr + (tile_row * config::SWIGLU_Nb + col_idx) * static_cast<int>(sizeof(bf16)),
                                          float2{__uint_as_float(d_gate_bf16_words[k * 2]), __uint_as_float(d_gate_bf16_words[k * 2 + 1])});
                        move<float2>::sts(d_up_bf16_addr + (tile_row * config::SWIGLU_Nb + col_idx) * static_cast<int>(sizeof(bf16)),
                                          float2{__uint_as_float(d_up_bf16_words[k * 2]), __uint_as_float(d_up_bf16_words[k * 2 + 1])});
                    }
                }

                // Store this thread's two scale bytes per gradient (overwrites the input scale tiles)
                const uint16_t d_gate_scale_pair = static_cast<uint16_t>(d_gate_scale_byte[tile_row & 1] | (d_gate_scale_byte[(tile_row + 1) & 1] << 8));
                const uint16_t d_up_scale_pair = static_cast<uint16_t>(d_up_scale_byte[tile_row & 1] | (d_up_scale_byte[(tile_row + 1) & 1] << 8));
                move<bf16>::sts(static_cast<uint32_t>(__cvta_generic_to_shared(&gate_sc_smem[stage])) + scale_word_offset + k_block_pair * 2, std::bit_cast<bf16>(d_gate_scale_pair));
                move<bf16>::sts(static_cast<uint32_t>(__cvta_generic_to_shared(&up_sc_smem[stage])) + scale_word_offset + k_block_pair * 2, std::bit_cast<bf16>(d_up_scale_pair));
                __syncthreads(); // all normal-orientation outputs and BF16 stagings must be visible

                if (threadIdx.x == 0) {
                    tma::store_async(d_gate_gmem, gate_smem[stage], {row - macrobatch_row_block_offset, col});
                    tma::store_async(*d_gate_sc_gmem, gate_sc_smem[stage], {row - macrobatch_row_block_offset, col, 0, 0});
                    tma::store_async(d_up_gmem, up_smem[stage], {row - macrobatch_row_block_offset, col});
                    tma::store_async(*d_up_sc_gmem, up_sc_smem[stage], {row - macrobatch_row_block_offset, col, 0, 0});
                }

                // Transpose-quantize both gradients, one per warpgroup; d_gate was staged in the d_hidden tile, d_up in the shared staging buffer
                if (threadIdx.x < config::QUANT_Mb)
                    mxfp8::quantize_tile<false, true>(d_hidden_smem[stage], d_t_fp8_smem[0], d_t_sc_smem[0], d_t_fp8_smem[0], d_t_sc_smem[0], nullptr, threadIdx.x, 1);
                else
                    mxfp8::quantize_tile<false, true>(d_up_bf16_smem, d_t_fp8_smem[1], d_t_sc_smem[1], d_t_fp8_smem[1], d_t_sc_smem[1], nullptr, threadIdx.x - config::QUANT_Mb, 2);
                
                // Store partial router gradient to smem for the first col half to handle later
                if (tile_col_half != 0) 
                    router_smem[stage][tile_row] = router_grad_partial;
                __syncthreads(); // quantized tiles must be complete before TMA reads them & partial router gradient must be fully stored

                if (threadIdx.x == 0) {
                    tma::store_async(*d_gate_t_gmem, d_t_fp8_smem[0], {col, row - macrobatch_row_block_offset});
                    tma::store_async(*d_gate_sc_t_gmem, d_t_sc_smem[0], {col, row - macrobatch_row_block_offset, 0, 0});
                    tma::store_async(*d_up_t_gmem, d_t_fp8_smem[1], {col, row - macrobatch_row_block_offset});
                    tma::store_async(*d_up_sc_t_gmem, d_t_sc_smem[1], {col, row - macrobatch_row_block_offset, 0, 0});
                    tma::store_async_read_wait(); // the next stage reuses the staging and transposed-output buffers
                }
                if (tile_col_half == 0 && peer_rank >= 0) {
                    router_grad_partial += router_smem[stage][tile_row];
                    (*d_router_weight_partials)[{local_token_idx, col}] = router_grad_partial;
                }
                __syncthreads();
            } else if constexpr (IS_SHARED) {
                if constexpr (USE_SITUGLU) {
                    const auto *d_hidden_pairs = reinterpret_cast<const bf16_2 *>(d_hidden_smem[stage].data);
                    auto *gate_pairs = reinterpret_cast<bf16_2 *>(gate_smem[stage].data);
                    auto *up_pairs = reinterpret_cast<bf16_2 *>(up_smem[stage].data);
                    #pragma unroll
                    for (int i = threadIdx.x; i < config::SWIGLU_Mb * config::SWIGLU_Nb / 2; i += config::NUM_THREADS) {
                        const float2 gate = __bfloat1622float2(gate_pairs[i]);
                        const float2 up = __bfloat1622float2(up_pairs[i]);
                        const float2 d_hidden = __bfloat1622float2(d_hidden_pairs[i]);
                        const situglu_backward_terms terms_x = situglu_backward(gate.x, up.x);
                        const situglu_backward_terms terms_y = situglu_backward(gate.y, up.y);
                        gate_pairs[i] = __floats2bfloat162_rn(
                            terms_x.d_gate * d_hidden.x,
                            terms_y.d_gate * d_hidden.y);
                        up_pairs[i] = __floats2bfloat162_rn(
                            terms_x.d_up * d_hidden.x,
                            terms_y.d_up * d_hidden.y);
                    }
                } else if constexpr (IS_CLAMPED) {
                    rt_bf<config::SWIGLU_Mb / config::NUM_WARPS, config::SWIGLU_Nb> gate, up, d_hidden;
                    compute_group::load(gate, gate_smem[stage]);
                    compute_group::load(up, up_smem[stage]);
                    compute_group::load(d_hidden, d_hidden_smem[stage]);
                    #pragma unroll
                    for (int i = 0; i < gate.height; ++i) {
                        #pragma unroll
                        for (int j = 0; j < gate.width; ++j) {
                            #pragma unroll
                            for (int k = 0; k < gate.packed_per_tile; ++k) {
                                const float2 gate_fp32 = __bfloat1622float2(gate.tiles[i][j].data[k]);
                                const float2 up_fp32 = __bfloat1622float2(up.tiles[i][j].data[k]);
                                const float2 d_hidden_fp32 = __bfloat1622float2(d_hidden.tiles[i][j].data[k]);
                                swiglu_bwd_pair<IS_CLAMPED>(gate_fp32, up_fp32, d_hidden_fp32, swiglu_limit, gate.tiles[i][j].data[k], up.tiles[i][j].data[k]);
                            }
                        }
                    }
                    compute_group::store(gate_smem[stage], gate);
                    compute_group::store(up_smem[stage], up);
                } else {
                    rt_fl<config::SWIGLU_Mb / config::NUM_WARPS, config::SWIGLU_Nb> gate, up, d_hidden;
                    compute_group::load(gate, gate_smem[stage]);
                    compute_group::mul(d_hidden, gate, -1.0f);
                    compute_group::exp(d_hidden, d_hidden);
                    compute_group::add(d_hidden, d_hidden, 1.0f);         // d_hidden := 1 / sigmoid(gate)
                    compute_group::div(gate, gate, d_hidden);             // gate := silu(gate)
                    compute_group::mul(up, gate, -1.0f);
                    compute_group::add(up, up, 1.0f);
                    compute_group::div(up, up, d_hidden);
                    compute_group::add(up, up, gate);                     // up := dsilu(gate)
                    compute_group::load(d_hidden, d_hidden_smem[stage]);
                    compute_group::mul(gate, gate, d_hidden);             // gate := d_up
                    compute_group::mul(d_hidden, d_hidden, up);
                    compute_group::load(up, up_smem[stage]);
                    compute_group::mul(d_hidden, d_hidden, up);           // d_hidden := d_gate
                    compute_group::store(gate_smem[stage], d_hidden);     // d_gate overwrites the gate tile in place
                    compute_group::store(up_smem[stage], gate);           // d_up overwrites the up tile in place
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    tma::store_async(d_gate_gmem, gate_smem[stage], {row - macrobatch_row_block_offset, col});
                    tma::store_async(d_up_gmem, up_smem[stage], {row - macrobatch_row_block_offset, col});
                }
            } else {
                const int tile_row = threadIdx.x % config::SWIGLU_Mb;
                const int tile_col_half = threadIdx.x / config::SWIGLU_Mb;
                const int global_token_idx = row * config::SWIGLU_Mb + tile_row;
                const int local_token_idx = (row - macrobatch_row_block_offset) * config::SWIGLU_Mb + tile_row;
                const int peer_rank = (*schedule_peer_rank)[{global_token_idx}];
                const float router_weight = router_weights->raw_ptr[local_token_idx];
                const float inv_router_weight = router_weight > 0.0f ? 1.0f / router_weight : 0.0f;
                float router_grad_partial = 0.0f;

                const uint32_t gate_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&gate_smem[stage]));
                const uint32_t up_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&up_smem[stage]));
                const uint32_t d_hidden_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&d_hidden_smem[stage]));
                #pragma unroll
                for (int i = 0; i < config::SWIGLU_Nb / 8; ++i) {
                    const int tile_col = tile_col_half * (config::SWIGLU_Nb / 2) + i * 4;
                    float2 gate_packed, up_packed, d_hidden_packed;
                    move<float2>::lds(gate_packed, swiglu_tile::idx(gate_addr, {tile_row, tile_col}));
                    move<float2>::lds(up_packed, swiglu_tile::idx(up_addr, {tile_row, tile_col}));
                    move<float2>::lds(d_hidden_packed, swiglu_tile::idx(d_hidden_addr, {tile_row, tile_col}));
                    const auto *gate_pairs = reinterpret_cast<const bf16_2 *>(&gate_packed);
                    const auto *up_pairs = reinterpret_cast<const bf16_2 *>(&up_packed);
                    const auto *d_hidden_pairs = reinterpret_cast<const bf16_2 *>(&d_hidden_packed);
                    bf16_2 d_gate_pairs[2], d_up_pairs[2];
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const float2 gate = __bfloat1622float2(gate_pairs[j]);
                        const float2 up = __bfloat1622float2(up_pairs[j]);
                        const float2 d_hidden = __bfloat1622float2(d_hidden_pairs[j]);
                        if constexpr (USE_SITUGLU) {
                            const situglu_backward_terms terms_x = situglu_backward(gate.x, up.x);
                            const situglu_backward_terms terms_y = situglu_backward(gate.y, up.y);
                            router_grad_partial += d_hidden.x * inv_router_weight * terms_x.hidden
                                                 + d_hidden.y * inv_router_weight * terms_y.hidden;
                            d_gate_pairs[j] = __floats2bfloat162_rn(
                                terms_x.d_gate * d_hidden.x,
                                terms_y.d_gate * d_hidden.y);
                            d_up_pairs[j] = __floats2bfloat162_rn(
                                terms_x.d_up * d_hidden.x,
                                terms_y.d_up * d_hidden.y);
                        } else {
                            const float2 hidden = swiglu_bwd_pair<IS_CLAMPED>(gate, up, d_hidden, swiglu_limit, d_gate_pairs[j], d_up_pairs[j]);
                            router_grad_partial += d_hidden.x * inv_router_weight * hidden.x
                                                 + d_hidden.y * inv_router_weight * hidden.y;
                        }
                    }
                    const auto *d_gate_words = reinterpret_cast<const uint32_t *>(d_gate_pairs);
                    const auto *d_up_words = reinterpret_cast<const uint32_t *>(d_up_pairs);
                    move<float2>::sts(swiglu_tile::idx(gate_addr, {tile_row, tile_col}),
                                      float2{__uint_as_float(d_gate_words[0]), __uint_as_float(d_gate_words[1])});
                    move<float2>::sts(swiglu_tile::idx(up_addr, {tile_row, tile_col}),
                                      float2{__uint_as_float(d_up_words[0]), __uint_as_float(d_up_words[1])});
                }

                if (tile_col_half != 0)
                    router_smem[stage][tile_row] = router_grad_partial;
                __syncthreads();
                if (tile_col_half == 0 && peer_rank >= 0) {
                    router_grad_partial += router_smem[stage][tile_row];
                    (*d_router_weight_partials)[{local_token_idx, col}] = router_grad_partial;
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    tma::store_async(d_gate_gmem, gate_smem[stage], {row - macrobatch_row_block_offset, col});
                    tma::store_async(d_up_gmem, up_smem[stage], {row - macrobatch_row_block_offset, col});
                }
            }
        }
    }

    if (threadIdx.x == 0) {
        tma::store_async_wait();
        #pragma unroll
        for (int stage = 0; stage < config::SWIGLU_BWD_PIPE_DEPTH; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < tile_end) {
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks)
                    ++row;
                barrier_arrive(d_gate_up_row_block_ready, d_gate_up_row_block_ready_base_index + row / (config::MLP_Mb / config::SWIGLU_Mb));
            }
        }
        if (buffer_done != nullptr)
            barrier_arrive(*buffer_done, buffer_done_index);
    }
}
