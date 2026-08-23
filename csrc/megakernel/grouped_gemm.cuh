template <bool IS_SHARED, bool IS_WGRAD = false, bool IS_AB = false>
static __device__ __forceinline__ void expert_grouped_gemm_kernel(
    const std::conditional_t<IS_SHARED, mlp_bf16_gl, routed_activation_gl> &a_gmem,
    const std::conditional_t<IS_WGRAD, std::conditional_t<IS_SHARED, wgrad_bf16_gl, routed_activation_gl>,
                                       std::conditional_t<IS_SHARED, weight_bf16_gl, routed_weight_gl>> &b_gmem,
    const routed_sc_gl *a_sc_gmem,          // routed MXFP8 only
    const routed_sc_gl *b_sc_gmem,          // routed MXFP8 only
    const std::conditional_t<IS_SHARED, mlp_bf16_gl, routed_activation_gl> *a2_gmem, // accumulated second GEMM
    const std::conditional_t<IS_SHARED, weight_bf16_gl, routed_weight_gl> *b2_gmem,
    const routed_sc_gl *a2_sc_gmem,         // routed MXFP8 only
    const routed_sc_gl *b2_sc_gmem,         // routed MXFP8 only
    const std::conditional_t<IS_WGRAD, d_weight_bf16_gl, epi_bf16_gl> &d_gmem,
    const routed_gate_up_gl *d_routed_gmem, // routed gate/up saved activation
    const routed_sc_gl *d_sc_gmem,          // routed MXFP8 only
    const index_gl &tokens_per_expert,
    const index_gl *input_minibatch_ready,  // comms -> GEMM
    const index_gl *input_row_block_ready,  // SwiGLU -> GEMM
    const index_gl *output_row_ready,       // previous combine -> GEMM epilogue
    const index_gl *output_tile_ready,      // GEMM -> SwiGLU
    const index_gl *output_minibatch_ready, // GEMM -> comms
    const index_gl *buffer_done,
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> &d_tt,
    const full_tt_fp8e8m0<16 * config::MLP_LOAD_PIPE_DEPTH> &a_sc_tt,
    const full_tt_fp8e8m0<32 * config::MLP_LOAD_PIPE_DEPTH> &b_sc_tt,
    semaphore (&gemm_inputs_arrived)[config::MLP_LOAD_PIPE_DEPTH],
    semaphore (&gemm_scales_arrived)[config::MLP_LOAD_PIPE_DEPTH],
    semaphore (&gemm_inputs_finished)[config::MLP_LOAD_PIPE_DEPTH],
    semaphore (&gemm_scales_finished)[config::MLP_LOAD_PIPE_DEPTH],
    semaphore &gemm_outputs_arrived,
    semaphore &gemm_outputs_finished,
    uint32_t &gemm_bitfield,
    const int num_tokens,
    const int macrobatch_size,
    const int minibatch_size,
    const int macrobatch_idx,
    const int minibatch_idx,
    int task_idx,
    const int cta_rank,
    const int input_minibatch_ready_num_cols,
    const int input_row_block_ready_base_index,
    const int input_row_block_ready_required_count,
    const int output_tile_ready_base_index,
    const int buffer_done_index,
    const uint64_t smem_base_addr
) {
    static constexpr bool USE_ROUTED_MXFP8 = !IS_SHARED && USE_MXFP8;
    using a_tile = std::conditional_t<USE_ROUTED_MXFP8, mlp_fp8_tile,
                                      std::conditional_t<IS_WGRAD, mlp_bf16_t_tile, mlp_bf16_tile>>;
    using b_tile = std::conditional_t<USE_ROUTED_MXFP8, mlp_fp8_tile,
                                      std::conditional_t<IS_WGRAD || IS_AB, mlp_bf16_t_tile, mlp_bf16_tile>>;
    constexpr int MLP_Kb = USE_ROUTED_MXFP8 ? config::MLP_FP8_Kb : config::MLP_BF16_Kb;

    auto (&a_smem)[config::MLP_LOAD_PIPE_DEPTH]       = *reinterpret_cast<a_tile (*)[config::MLP_LOAD_PIPE_DEPTH]>(smem_base_addr);
    auto (&b_smem)[config::MLP_LOAD_PIPE_DEPTH]       = *reinterpret_cast<b_tile (*)[config::MLP_LOAD_PIPE_DEPTH]>(smem_base_addr + sizeof(a_smem));
    auto (&a_sc_smem)[config::MLP_LOAD_PIPE_DEPTH]    = *reinterpret_cast<mlp_sc_tile (*)[config::MLP_LOAD_PIPE_DEPTH]>(smem_base_addr + sizeof(a_smem) + sizeof(b_smem));
    auto (&b_sc_smem)[config::MLP_LOAD_PIPE_DEPTH][2] = *reinterpret_cast<mlp_sc_tile (*)[config::MLP_LOAD_PIPE_DEPTH][2]>(smem_base_addr + sizeof(a_smem) + sizeof(b_smem) + sizeof(a_sc_smem));
    auto (&d_bf16_smem)[config::MLP_NUM_BF16_D_TILES] = *reinterpret_cast<mlp_bf16_d_tile (*)[config::MLP_NUM_BF16_D_TILES]>((smem_base_addr + sizeof(a_smem) + sizeof(b_smem) + sizeof(a_sc_smem) + sizeof(b_sc_smem) + 1023) & ~uint64_t(1023));
    auto &d_fp8_smem                                  = *reinterpret_cast<mlp_fp8_d_tile *>(&d_bf16_smem[2]);
    auto (&d_sc_smem)[2]                              = *reinterpret_cast<mlp_sc_tile (*)[2]>(reinterpret_cast<uint64_t>(&d_fp8_smem) + sizeof(d_fp8_smem));
    static_assert(config::MLP_NUM_BF16_D_TILES >= 3);
    static_assert(sizeof(mlp_fp8_d_tile) + 2 * sizeof(mlp_sc_tile) <= sizeof(mlp_bf16_d_tile));

    const int col_blocks = ((IS_WGRAD && !USE_ROUTED_MXFP8) || IS_AB) ? b_gmem.cols() / config::MLP_Nb : b_gmem.rows() / config::MLP_Nb;
    const int global_minibatch_idx = macrobatch_idx * (macrobatch_size / minibatch_size) + minibatch_idx;
    const int macrobatch_row_block_offset = macrobatch_idx * (macrobatch_size / config::MLP_Mb);

    // Output tile (and for wgrad, K range in token rows) of this task
    int3 tile_coord = {-1, -1, -1};
    int k_start = 0, k_end = 0;
    bool is_first_wgrad_contribution = IS_SHARED;
    if constexpr (IS_WGRAD) {
        const int row_blocks = USE_ROUTED_MXFP8 ? a_gmem.rows() / config::MLP_Mb : a_gmem.cols() / config::MLP_Mb;
        const int expert_idx = IS_SHARED ? 0 : task_idx / (row_blocks * col_blocks);
        if constexpr (IS_SHARED) {
            k_end = a_gmem.rows();
        } else {
            int expert_row_offset = 0;
            for (int i = 0; i < expert_idx; ++i)
                expert_row_offset += tokens_per_expert[{i}];
            k_start = max(expert_row_offset, macrobatch_idx * macrobatch_size);
            k_end = min(expert_row_offset + tokens_per_expert[{expert_idx}], min((macrobatch_idx + 1) * macrobatch_size, num_tokens));
            is_first_wgrad_contribution = k_start == expert_row_offset;
        }
        if (k_start < k_end) {
            const int2 swizzled = get_swizzled_2d_idx<config::MLP_SUPERGROUP_SIZE>(row_blocks, col_blocks, IS_SHARED ? task_idx : task_idx % (row_blocks * col_blocks));
            tile_coord = {swizzled.x, swizzled.y, expert_idx};
        }
    } else if constexpr (IS_SHARED) {
        const int row_blocks = a_gmem.rows() / config::MLP_Mb;
        const int num_tasks = row_blocks * col_blocks;
        if (task_idx < num_tasks) {
            const int2 swizzled = get_swizzled_2d_idx<config::MLP_SUPERGROUP_SIZE>(row_blocks, col_blocks, task_idx);
            tile_coord = {swizzled.x, swizzled.y, 0};
        }
    } else {
        const int minibatch_routed_row_blocks = minibatch_size / config::MLP_Mb;
        const int global_minibatch_routed_first_row_block = global_minibatch_idx * minibatch_routed_row_blocks;
        int global_row_block_offset = 0;
        for (int expert_idx = 0; expert_idx < b_gmem.depth(); ++expert_idx) {
            const int expert_row_blocks = tokens_per_expert[{expert_idx}] / config::MLP_Mb;
            const int global_first_row_block = max(global_minibatch_routed_first_row_block, global_row_block_offset);
            const int row_blocks = max(0, min(global_minibatch_routed_first_row_block + minibatch_routed_row_blocks, global_row_block_offset + expert_row_blocks) - global_first_row_block);
            const int num_tasks = row_blocks * col_blocks;
            if (task_idx < num_tasks) {
                const int2 swizzled = get_swizzled_2d_idx<config::MLP_SUPERGROUP_SIZE>(row_blocks, col_blocks, task_idx);
                tile_coord = {global_first_row_block + swizzled.x - macrobatch_row_block_offset, swizzled.y, expert_idx};
                break;
            }
            task_idx -= num_tasks;
            global_row_block_offset += expert_row_blocks;
        }
    }
    if (tile_coord.z < 0) {
        if (buffer_done != nullptr && threadIdx.x == 0)
            barrier_arrive(*buffer_done, buffer_done_index);
        return;
    }

    const int first_gemm_iters = IS_WGRAD ? 0 : a_gmem.cols() / MLP_Kb;
    const int iters_per_task = IS_WGRAD ? (k_end - k_start) / MLP_Kb
                                        : first_gemm_iters + (a2_gmem != nullptr ? a2_gmem->cols() / MLP_Kb : 0);

    auto wait_for_a_operand = [&]() {
        if (input_row_block_ready != nullptr) {
            barrier_wait(*input_row_block_ready, input_row_block_ready_base_index + macrobatch_row_block_offset + tile_coord.x, input_row_block_ready_required_count);
        }
        if (input_minibatch_ready != nullptr) {
            const int minibatch_first_row = global_minibatch_idx * minibatch_size;
            const int minibatch_rows = max(0, min(minibatch_size, num_tokens - minibatch_first_row));
            const int required_count = ((minibatch_rows + config::DISPATCH_Mb - 1) / config::DISPATCH_Mb) * ((a_gmem.cols() + config::DISPATCH_Nb - 1) / config::DISPATCH_Nb);
            barrier_wait(*input_minibatch_ready, global_minibatch_idx, required_count);
        }
    };
    // Wgrad iterates K over token rows: wait as the loads cross into each producing row block / minibatch
    auto wait_for_wgrad_operands = [&](int idx, int row) {
        if (idx == 0 || row % config::MLP_Mb == 0) {
            if (input_row_block_ready != nullptr)
                barrier_wait(*input_row_block_ready, input_row_block_ready_base_index + row / config::MLP_Mb, input_row_block_ready_required_count);
        }
        if (idx == 0 || row % minibatch_size == 0) {
            if (input_minibatch_ready != nullptr) {
                const int row_minibatch_idx = row / minibatch_size;
                const int minibatch_rows = min(minibatch_size, num_tokens - row_minibatch_idx * minibatch_size);
                const int required_count = ((minibatch_rows + config::DISPATCH_Mb - 1) / config::DISPATCH_Mb) * ((input_minibatch_ready_num_cols + config::DISPATCH_Nb - 1) / config::DISPATCH_Nb);
                barrier_wait(*input_minibatch_ready, row_minibatch_idx, required_count);
            }
        }
    };

    if (warpgroup::groupid() == config::NUM_CONSUMERS) {
        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            int input_ring = 0;
            if constexpr (IS_WGRAD) {
                const int macrobatch_k_offset = macrobatch_idx * (macrobatch_size / MLP_Kb);
                for (int idx = 0, k_block = k_start / MLP_Kb; idx < iters_per_task; ++idx, ++k_block) {
                    wait_for_wgrad_operands(idx, k_block * MLP_Kb);
                    wait(gemm_inputs_finished[input_ring], get_phasebit<1>(gemm_bitfield, input_ring));
                    if constexpr (!USE_ROUTED_MXFP8) { // BF16 AtB: A/B tiles are K-major slices of normal token-major activations
                        tma::cluster::load_async(a_smem[input_ring], a_gmem, {k_block - macrobatch_k_offset, tile_coord.x * 2 + cta_rank}, gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                        tma::cluster::load_async(b_smem[input_ring], b_gmem, {k_block - macrobatch_k_offset, tile_coord.y * 2 + cta_rank}, gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    } else { // MXFP8 ABt: A/B are transpose-quantized activations with K = tokens
                        tma::cluster::load_async(a_smem[input_ring], a_gmem, {tile_coord.x * 2 + cta_rank, k_block - macrobatch_k_offset}, gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                        tma::cluster::load_async(b_smem[input_ring], b_gmem, {tile_coord.y * 2 + cta_rank, k_block - macrobatch_k_offset}, gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    }
                    update_phasebit<1>(gemm_bitfield, input_ring);
                    input_ring = ring_advance<config::MLP_LOAD_PIPE_DEPTH>(input_ring);
                }
            } else {
                wait_for_a_operand();
                for (int idx = 0; idx < iters_per_task; ++idx) {
                    const auto &a_gmem_curr = idx < first_gemm_iters ? a_gmem : *a2_gmem;
                    const auto &b_gmem_curr = idx < first_gemm_iters ? b_gmem : *b2_gmem;
                    const int k_block = idx < first_gemm_iters ? idx : idx - first_gemm_iters;
                    wait(gemm_inputs_finished[input_ring], get_phasebit<1>(gemm_bitfield, input_ring));
                    tma::cluster::load_async(a_smem[input_ring], a_gmem_curr, {tile_coord.x * 2 + cta_rank, k_block},               gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    if constexpr (IS_AB)
                        tma::cluster::load_async(b_smem[input_ring], b_gmem_curr, {tile_coord.z, k_block, tile_coord.y * 2 + cta_rank}, gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    else
                        tma::cluster::load_async(b_smem[input_ring], b_gmem_curr, {tile_coord.z, tile_coord.y * 2 + cta_rank, k_block}, gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    update_phasebit<1>(gemm_bitfield, input_ring);
                    input_ring = ring_advance<config::MLP_LOAD_PIPE_DEPTH>(input_ring);
                }
            }
        } else if (warpgroup::warpid() == 2 && warp::elect_leader()) {
            if constexpr (USE_ROUTED_MXFP8) {
                int input_ring = 0;
                if constexpr (IS_WGRAD) { // MXFP8 wgrad: both operands' scales follow the token-major layout
                    const int macrobatch_k_offset = macrobatch_idx * (macrobatch_size / MLP_Kb);
                    for (int idx = 0, k_block = k_start / MLP_Kb; idx < iters_per_task; ++idx, ++k_block) {
                        wait_for_wgrad_operands(idx, k_block * MLP_Kb);
                        wait(gemm_scales_finished[input_ring], get_phasebit<1>(gemm_bitfield, input_ring));
                        tma::cluster::load_async(a_sc_smem[input_ring], *a_sc_gmem, {tile_coord.x * 2 + cta_rank, k_block - macrobatch_k_offset, 0, 0}, gemm_scales_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                        tma::cluster::load_async(b_sc_smem[input_ring][cta_rank], *b_sc_gmem, {tile_coord.y * 2 + cta_rank, k_block - macrobatch_k_offset, 0, 0}, gemm_scales_arrived[input_ring], (uint16_t)(0b11), 0);
                        update_phasebit<1>(gemm_bitfield, input_ring);
                        input_ring = ring_advance<config::MLP_LOAD_PIPE_DEPTH>(input_ring);
                    }
                } else {
                    wait_for_a_operand();
                    for (int idx = 0; idx < iters_per_task; ++idx) {
                        wait(gemm_scales_finished[input_ring], get_phasebit<1>(gemm_bitfield, input_ring));
                        const sc_gl &a_sc_curr = idx < first_gemm_iters ? *a_sc_gmem : *a2_sc_gmem;
                        const sc_gl &b_sc_curr = idx < first_gemm_iters ? *b_sc_gmem : *b2_sc_gmem;
                        const auto &b_gmem_curr = idx < first_gemm_iters ? b_gmem : *b2_gmem;
                        const int k_block = idx < first_gemm_iters ? idx : idx - first_gemm_iters;
                        tma::cluster::load_async(a_sc_smem[input_ring], a_sc_curr, {tile_coord.x * 2 + cta_rank, k_block, 0, 0}, gemm_scales_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                        tma::cluster::load_async(b_sc_smem[input_ring][cta_rank], b_sc_curr, {tile_coord.z * (b_gmem_curr.rows() / config::QUANT_Mb) + tile_coord.y * 2 + cta_rank, k_block, 0, 0}, gemm_scales_arrived[input_ring], (uint16_t)(0b11), 0);
                        update_phasebit<1>(gemm_bitfield, input_ring);
                        input_ring = ring_advance<config::MLP_LOAD_PIPE_DEPTH>(input_ring);
                    }
                }
            }
        } else if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
            int input_ring = 0;
            wait(gemm_outputs_finished, get_phasebit<1>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH));
            update_phasebit<1>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH);
            tensor_after_thread_sync();
            for (int idx = 0; idx < iters_per_task; ++idx) {
                if constexpr (USE_ROUTED_MXFP8) {
                    tma::expect_bytes(gemm_scales_arrived[input_ring], config::CLUSTER_SIZE * 3 * sizeof(mlp_sc_tile));
                    wait(gemm_scales_arrived[input_ring], get_phasebit<0>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH + 1 + input_ring));
                    update_phasebit<0>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH + 1 + input_ring);
                    auto a_sc_tt_subtile = a_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 16);
                    auto b_sc_tt_subtile_0 = b_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 32);
                    auto b_sc_tt_subtile_1 = b_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 32 + 16);
                    load_mxnv_scale_async2(a_sc_tt_subtile, a_sc_smem[input_ring]);
                    load_mxnv_scale_async2(b_sc_tt_subtile_0, b_sc_smem[input_ring][0]);
                    load_mxnv_scale_async2(b_sc_tt_subtile_1, b_sc_smem[input_ring][1], gemm_scales_finished[input_ring]);
                    tma::expect_bytes(gemm_inputs_arrived[input_ring], config::CLUSTER_SIZE * (sizeof(a_tile) + sizeof(b_tile)));
                    wait(gemm_inputs_arrived[input_ring], get_phasebit<0>(gemm_bitfield, input_ring));
                    if (idx == 0) mm2_ABt (d_tt, a_smem[input_ring], b_smem[input_ring],
                                           a_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 16),
                                           b_sc_tt.template subtile<full_tt_fp8e8m0<32>>(input_ring * 32),
                                           gemm_inputs_finished[input_ring]);
                    else          mma2_ABt(d_tt, a_smem[input_ring], b_smem[input_ring],
                                           a_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 16),
                                           b_sc_tt.template subtile<full_tt_fp8e8m0<32>>(input_ring * 32),
                                           gemm_inputs_finished[input_ring]);
                } else {
                    tma::expect_bytes(gemm_inputs_arrived[input_ring], config::CLUSTER_SIZE * (sizeof(a_tile) + sizeof(b_tile)));
                    wait(gemm_inputs_arrived[input_ring], get_phasebit<0>(gemm_bitfield, input_ring));
                    if constexpr (IS_WGRAD) {
                        if (idx == 0) mm2_AtB (d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                        else          mma2_AtB(d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                    } else if constexpr (IS_AB) {
                        if (idx == 0) mm2_AB (d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                        else          mma2_AB(d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                    } else {
                        if (idx == 0) mm2_ABt (d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                        else          mma2_ABt(d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                    }
                }
                update_phasebit<0>(gemm_bitfield, input_ring);
                input_ring = ring_advance<config::MLP_LOAD_PIPE_DEPTH>(input_ring);
            }
            detail::tcgen05::commit<config::CLUSTER_SIZE>(gemm_outputs_arrived);
        }
    } else {
        using epilogue_group = group<WARPGROUP_WARPS>;
        wait(gemm_outputs_arrived, get_phasebit<0>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH));
        update_phasebit<0>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH);
        auto store_bf16 = [&]() {
            rt_bf<config::MLP_Mb / 8, config::MLP_Nb / config::MLP_EPI_PIPE_DEPTH> d_reg[config::MLP_EPI_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < config::MLP_EPI_PIPE_DEPTH; ++i)
                warpgroup::load_async(d_reg[i], d_tt.template subtile<tt<float, config::MLP_Mb / 2, config::MLP_Nb / config::MLP_EPI_PIPE_DEPTH>>(0, config::MLP_Nb / config::MLP_EPI_PIPE_DEPTH * i));
            tensor_load_wait();
            warpgroup::sync(1);
            warpgroup::tma::cluster::arrive(gemm_outputs_finished, 0);
            if (output_row_ready != nullptr && epilogue_group::laneid() == 0) {
                const int previous_macrobatch_offset = (macrobatch_idx + 1) * macrobatch_size;
                const int row_idx = tile_coord.x * config::MLP_Mb + cta_rank * (config::MLP_Mb / config::CLUSTER_SIZE);
                const int previous_macrobatch_tokens = min(macrobatch_size, num_tokens - previous_macrobatch_offset);
                const int required_count = (config::MLP_Mb / config::CLUSTER_SIZE / config::COMBINE_Mb) * ((d_gmem.cols() + config::COMBINE_Nb - 1) / config::COMBINE_Nb);
                if (row_idx < previous_macrobatch_tokens)
                    barrier_wait(*output_row_ready, (previous_macrobatch_offset + row_idx) / (config::MLP_Mb / config::CLUSTER_SIZE), required_count);
            }
            #pragma unroll
            for (int i = 0; i < config::MLP_EPI_PIPE_DEPTH; ++i) {
                warpgroup::tma::store_async_read_wait<config::MLP_NUM_BF16_D_TILES - 1>();
                warpgroup::sync(1);
                warpgroup::store(d_bf16_smem[i % config::MLP_NUM_BF16_D_TILES], d_reg[i]);
                warpgroup::sync(1);
                if constexpr (IS_WGRAD) {
                    if (is_first_wgrad_contribution)
                        warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(d_gmem, d_bf16_smem[i % config::MLP_NUM_BF16_D_TILES], {tile_coord.z, 2 * tile_coord.x + cta_rank, config::MLP_EPI_PIPE_DEPTH * tile_coord.y + i});
                    else
                        // Macrobatches are serialized by routed_buffers_done, so additions occur in a fixed order, preserving determinism
                        warpgroup::tma::store_add_async<dim::ROW, cache_policy::EVICT_FIRST>(d_gmem, d_bf16_smem[i % config::MLP_NUM_BF16_D_TILES], {tile_coord.z, 2 * tile_coord.x + cta_rank, config::MLP_EPI_PIPE_DEPTH * tile_coord.y + i});
                } else {
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(d_gmem, d_bf16_smem[i % config::MLP_NUM_BF16_D_TILES], {2 * tile_coord.x + cta_rank, config::MLP_EPI_PIPE_DEPTH * tile_coord.y + i});
                }
            }
            warpgroup::tma::store_async_read_wait();
        };
        if constexpr (USE_ROUTED_MXFP8) {
          if (d_routed_gmem != nullptr) {
            constexpr int NUM_MXFP8_BLOCKS = config::MLP_Nb / 32;
            const int tile_row = warpgroup::laneid();
            uint32_t scale_word = 0;
            #pragma unroll 1
            for (int i = 0; i < NUM_MXFP8_BLOCKS; ++i) {
                float2 tmp[16];
                asm volatile(R"(
                    tcgen05.ld.sync.aligned.32x32b.x32.b32
                    {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,
                     %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];
                    )"
                    : "=f"(tmp[0].x), "=f"(tmp[0].y), "=f"(tmp[1].x), "=f"(tmp[1].y),
                      "=f"(tmp[2].x), "=f"(tmp[2].y), "=f"(tmp[3].x), "=f"(tmp[3].y),
                      "=f"(tmp[4].x), "=f"(tmp[4].y), "=f"(tmp[5].x), "=f"(tmp[5].y),
                      "=f"(tmp[6].x), "=f"(tmp[6].y), "=f"(tmp[7].x), "=f"(tmp[7].y),
                      "=f"(tmp[8].x), "=f"(tmp[8].y), "=f"(tmp[9].x), "=f"(tmp[9].y),
                      "=f"(tmp[10].x), "=f"(tmp[10].y), "=f"(tmp[11].x), "=f"(tmp[11].y),
                      "=f"(tmp[12].x), "=f"(tmp[12].y), "=f"(tmp[13].x), "=f"(tmp[13].y),
                      "=f"(tmp[14].x), "=f"(tmp[14].y), "=f"(tmp[15].x), "=f"(tmp[15].y)
                    : "r"(d_tt.addr + ((warpgroup::warpid() * 32) << 16) + i * 32));
                tensor_load_wait();
                bf16_2 d_reg[16];
                #pragma unroll
                for (int j = 0; j < 16; ++j)
                    d_reg[j] = __float22bfloat162_rn(tmp[j]);
                warpgroup::sync(1);
                const uint32_t d_bf16_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&d_bf16_smem[i % 2]));
                const uint32_t *d_bf16_words = reinterpret_cast<const uint32_t *>(d_reg);
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    move<float4>::sts(mlp_bf16_d_tile::idx(d_bf16_addr, {tile_row, j * 8}),
                        float4{__uint_as_float(d_bf16_words[j * 4]), __uint_as_float(d_bf16_words[j * 4 + 1]),
                               __uint_as_float(d_bf16_words[j * 4 + 2]), __uint_as_float(d_bf16_words[j * 4 + 3])});
                warpgroup::sync(1);
                warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(d_gmem, d_bf16_smem[i % 2], {2 * tile_coord.x + cta_rank, NUM_MXFP8_BLOCKS * tile_coord.y + i});

                // Wait for previous iteration's FP8 store to complete
                warpgroup::tma::store_async_read_wait<1>();
                warpgroup::sync(1);
                uint32_t d_fp8[8];
                uint32_t d_scale_byte;
                mxfp8::quantize_single_block(d_reg, d_fp8, d_scale_byte);
                scale_word |= d_scale_byte << ((i % 4) * 8);
                const uint32_t d_fp8_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&d_fp8_smem));
                #pragma unroll
                for (int m = 0; m < 2; ++m) {
                    move<float4>::sts(mlp_fp8_d_tile::idx(d_fp8_addr, {tile_row, m * 16}),
                        float4{__uint_as_float(d_fp8[m * 4]), __uint_as_float(d_fp8[m * 4 + 1]), __uint_as_float(d_fp8[m * 4 + 2]), __uint_as_float(d_fp8[m * 4 + 3])});
                }
                if (i % 4 == 3) {
                    move<int>::sts(static_cast<uint32_t>(__cvta_generic_to_shared(&d_sc_smem[i / 4])) + (tile_row % 32) * 16 + (tile_row / 32) * 4, std::bit_cast<int>(scale_word));
                    scale_word = 0;
                }
                warpgroup::sync(1);
                if (warpgroup::laneid() == 0) {
                    tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(*d_routed_gmem, d_fp8_smem, {2 * tile_coord.x + cta_rank, NUM_MXFP8_BLOCKS * tile_coord.y + i});
                    if (i % 4 == 3)
                        tma::store_async(*d_sc_gmem, d_sc_smem[i / 4], {2 * tile_coord.x + cta_rank, 2 * tile_coord.y + i / 4, 0, 0});
                }
            }
            tensor_before_thread_sync();
            warpgroup::sync(1);
            warpgroup::tma::cluster::arrive(gemm_outputs_finished, 0);
          } else {
            store_bf16();
          }
        } else {
            store_bf16();
        }
        epilogue_group::sync(4);
        if (epilogue_group::warpid() == 0 && warp::elect_leader()) {
            if (output_tile_ready != nullptr) {
                tma::store_async_wait();
                barrier_arrive(*output_tile_ready, output_tile_ready_base_index + (macrobatch_row_block_offset + tile_coord.x) * col_blocks + tile_coord.y);
            } else if (output_minibatch_ready != nullptr) {
                tma::store_async_wait();
                barrier_arrive(*output_minibatch_ready, global_minibatch_idx);
            }
            if (buffer_done != nullptr) {
                tma::store_async_wait();
                barrier_arrive(*buffer_done, buffer_done_index);
            }
        }
    }
}
