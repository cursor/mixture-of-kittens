template <bool IS_CLAMPED>
static __device__ __forceinline__ void recompute_forward_context_kernel(const globals_recompute_forward_context &g) {
    int cluster_idx = clusterIdx().x;
    const int cta_rank = cluster_ctarank();
    const int shared_row_blocks = g.x_shared.rows() / config::MLP_Mb;
    const int minibatch_routed_row_blocks = g.minibatch_size / config::MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (g.w_shared_gate.rows() / config::MLP_Nb);
    const int minibatch_routed_gate_up_tasks = minibatch_routed_row_blocks * (g.w_routed_gate.rows() / config::MLP_Nb);
    const int shared_swiglu_tiles = (g.hidden_shared.rows() / config::SWIGLU_Mb) * (g.hidden_shared.cols() / config::SWIGLU_Nb);
    const int minibatch_routed_swiglu_tiles = (g.minibatch_size / config::SWIGLU_Mb) * (g.hidden_fp8_routed.cols() / config::SWIGLU_Nb);
    const int shared_swiglu_tasks = (shared_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH);
    const int minibatch_routed_swiglu_tasks = (minibatch_routed_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH);
    const int shared_tasks = 2 * shared_gate_up_tasks + shared_swiglu_tasks;
    const int minibatch_tasks = 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_tasks;
    const int comm_clusters = g.num_comm_sms / config::CLUSTER_SIZE;
    const int num_tokens = g.num_tokens[{0}];
    const int routed_num_tokens = min(num_tokens, g.macrobatch_size);
    const int num_routed_minibatches = (routed_num_tokens + g.minibatch_size - 1) / g.minibatch_size;
    const int true_num_clusters = comm_clusters + shared_tasks + num_routed_minibatches * minibatch_tasks;
    if (cluster_idx >= true_num_clusters) return;

    warpgroup::increase_registers<256>();

    extern __shared__ int __shm[];
    const uint64_t smem_base_addr = (reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023);

    uint32_t gemm_bitfield = 0xFFFF0000;
    uint32_t swiglu_bitfield = 0xFFFF0000;
    uint32_t dispatch_bitfield = 0xFFFF0000;

    __shared__ clc::handle clc_handle[config::CLC_PIPE_DEPTH];
    __shared__ clc::handle clc_drain_handle[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore schedule_arrived[config::CLC_PIPE_DEPTH], schedule_finished[config::CLC_PIPE_DEPTH];
    __shared__ semaphore drain_schedule_arrived[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore drain_schedule_finished[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore swiglu_inputs_arrived[config::SWIGLU_FWD_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_arrived[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_scales_arrived[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_finished[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_scales_finished[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_outputs_arrived, gemm_outputs_finished;
    __shared__ semaphore dispatch_inputs_arrived;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < config::SWIGLU_FWD_PIPE_DEPTH; ++i) {
            init_semaphore(swiglu_inputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < config::MLP_LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(gemm_inputs_arrived[i], 0, 1);
            init_semaphore(gemm_scales_arrived[i], 0, 1);
            init_semaphore(gemm_inputs_finished[i], 0, 1);
            init_semaphore(gemm_scales_finished[i], 0, 1);
        }
        init_semaphore(gemm_outputs_arrived, 0, 1);
        init_semaphore(gemm_outputs_finished, 0, config::CLUSTER_SIZE);
        #pragma unroll
        for (int i = 0; i < config::CLC_PIPE_DEPTH; ++i) {
            init_semaphore(schedule_arrived[i], 0, 1);
            init_semaphore(schedule_finished[i], 0, config::CLUSTER_SIZE * config::NUM_WARPS);
        }
        #pragma unroll
        for (int i = 0; i < config::CLC_DRAIN_PIPE_DEPTH; ++i) {
            init_semaphore(drain_schedule_arrived[i], 0, 1);
            init_semaphore(drain_schedule_finished[i], 0, config::CLUSTER_SIZE);
        }
        init_semaphore(dispatch_inputs_arrived, 0, 1);
    }

    tensor_allocator<1, config::CLUSTER_SIZE> tm_alloc{};
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> d_tt = tm_alloc.template allocate<tt<float, config::MLP_Mb / 2, config::MLP_Nb>>(0);
    full_tt_fp8e8m0<16 * config::MLP_LOAD_PIPE_DEPTH> a_sc_tt = tm_alloc.template allocate<full_tt_fp8e8m0<16 * config::MLP_LOAD_PIPE_DEPTH>>(256);
    full_tt_fp8e8m0<32 * config::MLP_LOAD_PIPE_DEPTH> b_sc_tt = tm_alloc.template allocate<full_tt_fp8e8m0<32 * config::MLP_LOAD_PIPE_DEPTH>>(384);
    everyone::tma::cluster::sync();

    if (cluster_idx < comm_clusters) {
        const int comm_cta_idx = cluster_idx * config::CLUSTER_SIZE + cta_rank;
        const int dispatch_col_blocks = (g.x_fp8_routed.cols() + config::DISPATCH_Nb - 1) / config::DISPATCH_Nb;
        const int num_dispatch_tasks = (routed_num_tokens / config::DISPATCH_Mb) * dispatch_col_blocks;
        for (int task_idx = comm_cta_idx; task_idx < num_dispatch_tasks; task_idx += g.num_comm_sms) {
            dispatch_kernel<false>(g.x_routed_send_buffer, g.x_fp8_routed, &g.x_sc_routed, &g.x_fp8_t_routed, &g.x_sc_t_routed,
                                   nullptr, g.schedule_peer_rank, g.schedule_peer_token_idx,
                                   nullptr, nullptr, g.x_routed_ready,
                                   dispatch_inputs_arrived, dispatch_bitfield,
                                   num_tokens, g.macrobatch_size, g.minibatch_size, 0, task_idx, g.topk,
                                   -1, 0, smem_base_addr);
        }
        return;
    }

    auto is_cta_local_task = [&](int compute_cluster_idx) {
        if (compute_cluster_idx < 0) return false;
        if (compute_cluster_idx < 2 * shared_gate_up_tasks) return false;
        if (compute_cluster_idx < shared_tasks) return true;
        const int minibatch_task_idx = (compute_cluster_idx - shared_tasks) % minibatch_tasks;
        return minibatch_task_idx >= 2 * minibatch_routed_gate_up_tasks;
    };

    for (int task_iter = 0; cluster_idx >= 0 && cluster_idx < true_num_clusters; ++task_iter) {
        const int clc_stage = task_iter % config::CLC_PIPE_DEPTH;
        if (warpgroup::groupid() == config::NUM_CONSUMERS && warpgroup::warpid() == 1 && warp::elect_leader()) { // warp not used by the gemms
            if (cta_rank == 0) {
                wait(schedule_finished[clc_stage], ((task_iter + config::CLC_PIPE_DEPTH) / config::CLC_PIPE_DEPTH) % 2);
                clc::schedule(clc_handle[clc_stage], schedule_arrived[clc_stage]);
            }
            tma::expect_bytes(schedule_arrived[clc_stage], sizeof(clc_handle[clc_stage]));
        }

        const int compute_cluster_idx = cluster_idx - comm_clusters;
        const bool current_is_cta_local = is_cta_local_task(compute_cluster_idx);

        if (compute_cluster_idx < shared_gate_up_tasks) {
            // Shared gate (BF16)
            const int task_idx = compute_cluster_idx;
            expert_grouped_gemm_kernel<true>(g.x_shared, g.w_shared_gate, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                      g.gate_shared, nullptr, nullptr,
                                      g.tokens_per_expert, nullptr, nullptr, nullptr, &g.gate_up_tile_ready, nullptr, nullptr,
                                      d_tt, a_sc_tt, b_sc_tt,
                                      gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                      num_tokens, g.macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                      0, 0, 0, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_gate_up_tasks * 2) {
            // Shared up (BF16)
            const int task_idx = compute_cluster_idx - shared_gate_up_tasks;
            expert_grouped_gemm_kernel<true>(g.x_shared, g.w_shared_up, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                      g.up_shared, nullptr, nullptr,
                                      g.tokens_per_expert, nullptr, nullptr, nullptr, &g.gate_up_tile_ready, nullptr, nullptr,
                                      d_tt, a_sc_tt, b_sc_tt,
                                      gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                      num_tokens, g.macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                      0, 0, 0, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_tasks) {
            // Shared Swiglu (BF16)
            const int task_idx = compute_cluster_idx - shared_gate_up_tasks * 2;
            swiglu_fwd_kernel<true, IS_CLAMPED>(g.gate_shared, g.up_shared, g.hidden_shared, nullptr, nullptr, nullptr,
                             g.gate_up_tile_ready, g.hidden_row_block_ready,
                             swiglu_inputs_arrived, swiglu_bitfield,
                             g.x_shared.rows(), g.swiglu_limit, g.macrobatch_size, g.minibatch_size,
                             0, 0, task_idx, cta_rank, 0, 0, smem_base_addr);
        } else {
            const int routed_task_idx = compute_cluster_idx - shared_tasks;
            const int minibatch_idx = routed_task_idx / minibatch_tasks;
            const int minibatch_task_idx = routed_task_idx % minibatch_tasks;

            if (minibatch_task_idx < minibatch_routed_gate_up_tasks) {
                // Routed gate
                const int task_idx = minibatch_task_idx;
                expert_grouped_gemm_kernel<false>(g.x_fp8_routed, g.w_routed_gate, &g.x_sc_routed, &g.w_routed_gate_sc, nullptr, nullptr, nullptr, nullptr,
                                           g.gate_routed, &g.gate_fp8_routed, &g.gate_sc_routed,
                                           g.tokens_per_expert, &g.x_routed_ready, nullptr, nullptr, &g.gate_up_tile_ready, nullptr, nullptr,
                                           d_tt, a_sc_tt, b_sc_tt,
                                           gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                           num_tokens, g.macrobatch_size, g.minibatch_size, 0, minibatch_idx, task_idx, cta_rank,
                                           0, 0, 0, shared_gate_up_tasks, 0, smem_base_addr);
            } else if (minibatch_task_idx < minibatch_routed_gate_up_tasks * 2) {
                // Routed up
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks;
                expert_grouped_gemm_kernel<false>(g.x_fp8_routed, g.w_routed_up, &g.x_sc_routed, &g.w_routed_up_sc, nullptr, nullptr, nullptr, nullptr,
                                           g.up_routed, &g.up_fp8_routed, &g.up_sc_routed,
                                           g.tokens_per_expert, &g.x_routed_ready, nullptr, nullptr, &g.gate_up_tile_ready, nullptr, nullptr,
                                           d_tt, a_sc_tt, b_sc_tt,
                                           gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                           num_tokens, g.macrobatch_size, g.minibatch_size, 0, minibatch_idx, task_idx, cta_rank,
                                           0, 0, 0, shared_gate_up_tasks, 0, smem_base_addr);
            } else {
                // Routed Swiglu
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks * 2;
                swiglu_fwd_kernel<false, IS_CLAMPED>(g.gate_routed, g.up_routed, g.hidden_fp8_routed,
                                  &g.hidden_sc_routed, &g.hidden_fp8_t_routed, &g.hidden_sc_t_routed,
                                  g.gate_up_tile_ready, g.hidden_row_block_ready,
                                  swiglu_inputs_arrived, swiglu_bitfield,
                                  num_tokens, g.swiglu_limit, g.macrobatch_size, g.minibatch_size,
                                  0, minibatch_idx, task_idx, cta_rank,
                                  shared_gate_up_tasks, shared_row_blocks, smem_base_addr);
            }
        }

        wait(schedule_arrived[clc_stage], (task_iter / config::CLC_PIPE_DEPTH) % 2);
        const auto schedule = clc::query(clc_handle[clc_stage]);
        cluster_idx = schedule.success ? static_cast<int>(schedule.x / config::CLUSTER_SIZE) : -1;
        __syncwarp();
        warp::tma::cluster::arrive(schedule_finished[clc_stage], 0);

        // SWIGLU -> GEMM requires a cluster-wide sync
        const int next_compute_cluster_idx = cluster_idx - comm_clusters;
        if (current_is_cta_local && cluster_idx >= 0 && !is_cta_local_task(next_compute_cluster_idx))
            everyone::tma::cluster::sync();
    }

    everyone::tma::cluster::sync();

    // CLC drain for no-op threadblocks
    if (cluster_idx >= 0 && warp::laneid() == 0) {
        const int stage = warpid();
        int iter = 0;
        if (cta_rank == 0)
            clc::schedule(clc_drain_handle[stage], drain_schedule_arrived[stage]);
        tma::expect_bytes(drain_schedule_arrived[stage], sizeof(clc::handle));
        while (true) {
            wait(drain_schedule_arrived[stage], iter % 2);
            const auto schedule = clc::query(clc_drain_handle[stage]);
            warp::tma::cluster::arrive(drain_schedule_finished[stage], 0);
            if (cta_rank == 0)
                wait(drain_schedule_finished[stage], iter % 2);
            if (!schedule.success)
                break;
            if (cta_rank == 0)
                clc::schedule(clc_drain_handle[stage], drain_schedule_arrived[stage]);
            tma::expect_bytes(drain_schedule_arrived[stage], sizeof(clc::handle));
            ++iter;
        }
    }
}

static __host__ __forceinline__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                           at::Tensor>
recompute_forward_context_mxfp8(
    // Inputs and communication buffers
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,

    // Weights
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_routed_gate_sc,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_routed_up_sc,

    // Dispatch/combine schedule
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,

    // Metadata
    int topk,
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_local_tokens = x.size(0);
    const int schedule_capacity = schedule_peer_rank.size(0);
    const int num_routed_tokens = min(schedule_capacity, macrobatch_size);
    const int hidden_dim = x.size(1);
    const int intermediate_dim = w_shared_gate.size(0);
    const int num_routed_minibatches = (num_routed_tokens + minibatch_size - 1) / minibatch_size;
    const int shared_row_blocks = num_local_tokens / config::MLP_Mb;
    const int routed_row_blocks = num_routed_tokens / config::MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (w_shared_gate.size(0) / config::MLP_Nb);
    const int routed_gate_up_tasks = routed_row_blocks * (w_routed_gate.size(1) / config::MLP_Nb);

    activation_bf16_pgl x_routed_send_buffer_data;
    for (int i = 0; i < NUM_DEVICES; ++i)
        x_routed_send_buffer_data[i] = reinterpret_cast<bf16*>(x_ptrs[i]);

    at::Tensor x_fp8_routed = at::empty({macrobatch_size, hidden_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor x_sc_routed = at::empty({macrobatch_size / 128, hidden_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor x_fp8_t_routed = at::empty({hidden_dim, macrobatch_size}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor x_sc_t_routed = at::empty({hidden_dim / 128, macrobatch_size / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor gate_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor gate_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor gate_fp8_routed = at::empty({macrobatch_size, intermediate_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor gate_sc_routed = at::empty({macrobatch_size / 128, intermediate_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor up_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor up_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor up_fp8_routed = at::empty({macrobatch_size, intermediate_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor up_sc_routed = at::empty({macrobatch_size / 128, intermediate_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor hidden_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor hidden_fp8_routed = at::empty({macrobatch_size, intermediate_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor hidden_sc_routed = at::empty({macrobatch_size / 128, intermediate_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor hidden_fp8_t_routed = at::empty({intermediate_dim, macrobatch_size}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor hidden_sc_t_routed = at::empty({intermediate_dim / 128, macrobatch_size / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor x_routed_ready = at::zeros({num_routed_minibatches}, tokens_per_expert.options());
    at::Tensor gate_up_tile_ready = at::zeros({shared_gate_up_tasks + routed_gate_up_tasks}, tokens_per_expert.options());
    at::Tensor hidden_row_block_ready = at::zeros({shared_row_blocks + routed_row_blocks}, tokens_per_expert.options());

    globals_recompute_forward_context g {
        .x_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(x),
        .x_fp8_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(x_fp8_routed),
        .x_sc_routed = kittens::py::tensor_to_gl<sc_gl>(x_sc_routed),
        .x_fp8_t_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(x_fp8_t_routed),
        .x_sc_t_routed = kittens::py::tensor_to_gl<sc_gl>(x_sc_t_routed),
        .gate_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(gate_shared),
        .gate_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(gate_routed),
        .gate_fp8_routed = kittens::py::tensor_to_gl<gate_up_fp8_gl>(gate_fp8_routed),
        .gate_sc_routed = kittens::py::tensor_to_gl<sc_gl>(gate_sc_routed),
        .up_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(up_shared),
        .up_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(up_routed),
        .up_fp8_routed = kittens::py::tensor_to_gl<gate_up_fp8_gl>(up_fp8_routed),
        .up_sc_routed = kittens::py::tensor_to_gl<sc_gl>(up_sc_routed),
        .hidden_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(hidden_shared),
        .hidden_fp8_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(hidden_fp8_routed),
        .hidden_sc_routed = kittens::py::tensor_to_gl<sc_gl>(hidden_sc_routed),
        .hidden_fp8_t_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(hidden_fp8_t_routed),
        .hidden_sc_t_routed = kittens::py::tensor_to_gl<sc_gl>(hidden_sc_t_routed),
        .x_routed_send_buffer = x_routed_send_buffer_data,
        .w_shared_gate = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_gate),
        .w_routed_gate = kittens::py::tensor_to_gl<weight_fp8_gl>(w_routed_gate),
        .w_routed_gate_sc = kittens::py::tensor_to_gl<sc_gl>(w_routed_gate_sc),
        .w_shared_up = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_up),
        .w_routed_up = kittens::py::tensor_to_gl<weight_fp8_gl>(w_routed_up),
        .w_routed_up_sc = kittens::py::tensor_to_gl<sc_gl>(w_routed_up_sc),
        .schedule_peer_rank = kittens::py::tensor_to_gl<index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<index_gl>(schedule_peer_token_idx),
        .num_tokens = kittens::py::tensor_to_gl<index_gl>(num_tokens),
        .tokens_per_expert = kittens::py::tensor_to_gl<index_gl>(tokens_per_expert),
        .gate_up_tile_ready = kittens::py::tensor_to_gl<index_gl>(gate_up_tile_ready),
        .hidden_row_block_ready = kittens::py::tensor_to_gl<index_gl>(hidden_row_block_ready),
        .x_routed_ready = kittens::py::tensor_to_gl<index_gl>(x_routed_ready),
        .topk = topk,
        .swiglu_limit = swiglu_limit.value_or(0.0f),
        .num_comm_sms = num_comm_sms,
        .macrobatch_size = macrobatch_size,
        .minibatch_size = minibatch_size
    };

    if (swiglu_limit.has_value())
        kittens::py::launch_kernel<config, globals_recompute_forward_context, recompute_forward_context_kernel<true>>(g);
    else
        kittens::py::launch_kernel<config, globals_recompute_forward_context, recompute_forward_context_kernel<false>>(g);

    return {x_fp8_t_routed, x_sc_t_routed,
            gate_shared, gate_fp8_routed, gate_sc_routed,
            up_shared, up_fp8_routed, up_sc_routed,
            hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed};
}

static __host__ __forceinline__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                           at::Tensor, at::Tensor, at::Tensor>
recompute_forward_context_bf16(
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,
    int topk,
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    static_assert(!USE_MXFP8);
    const int num_local_tokens = x.size(0);
    const int schedule_capacity = schedule_peer_rank.size(0);
    const int num_routed_tokens = min(schedule_capacity, macrobatch_size);
    const int hidden_dim = x.size(1);
    const int intermediate_dim = w_shared_gate.size(0);
    const int num_routed_minibatches = (num_routed_tokens + minibatch_size - 1) / minibatch_size;
    const int shared_row_blocks = num_local_tokens / config::MLP_Mb;
    const int routed_row_blocks = num_routed_tokens / config::MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (intermediate_dim / config::MLP_Nb);
    const int routed_gate_up_tasks = routed_row_blocks * (intermediate_dim / config::MLP_Nb);

    activation_bf16_pgl x_routed_send_buffer_data;
    for (int i = 0; i < NUM_DEVICES; ++i)
        x_routed_send_buffer_data[i] = reinterpret_cast<bf16*>(x_ptrs[i]);

    at::Tensor x_routed = at::empty({macrobatch_size, hidden_dim}, x.options());
    at::Tensor gate_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor gate_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor up_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor up_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor hidden_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor hidden_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor x_routed_ready = at::zeros({num_routed_minibatches}, tokens_per_expert.options());
    at::Tensor gate_up_tile_ready = at::zeros({shared_gate_up_tasks + routed_gate_up_tasks}, tokens_per_expert.options());
    at::Tensor hidden_row_block_ready = at::zeros({shared_row_blocks + routed_row_blocks}, tokens_per_expert.options());

    globals_recompute_forward_context g {
        .x_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(x),
        .x_fp8_routed = kittens::py::tensor_to_gl<routed_activation_gl>(x_routed),
        .x_sc_routed = {},
        .x_fp8_t_routed = kittens::py::tensor_to_gl<routed_transposed_gl>(x_routed),
        .x_sc_t_routed = {},
        .gate_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(gate_shared),
        .gate_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(gate_routed),
        .gate_fp8_routed = kittens::py::tensor_to_gl<routed_gate_up_gl>(gate_routed),
        .gate_sc_routed = {},
        .up_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(up_shared),
        .up_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(up_routed),
        .up_fp8_routed = kittens::py::tensor_to_gl<routed_gate_up_gl>(up_routed),
        .up_sc_routed = {},
        .hidden_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(hidden_shared),
        .hidden_fp8_routed = kittens::py::tensor_to_gl<routed_activation_gl>(hidden_routed),
        .hidden_sc_routed = {},
        .hidden_fp8_t_routed = kittens::py::tensor_to_gl<routed_transposed_gl>(hidden_routed),
        .hidden_sc_t_routed = {},
        .x_routed_send_buffer = x_routed_send_buffer_data,
        .w_shared_gate = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_gate),
        .w_routed_gate = kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_gate),
        .w_routed_gate_sc = {},
        .w_shared_up = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_up),
        .w_routed_up = kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_up),
        .w_routed_up_sc = {},
        .schedule_peer_rank = kittens::py::tensor_to_gl<index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<index_gl>(schedule_peer_token_idx),
        .num_tokens = kittens::py::tensor_to_gl<index_gl>(num_tokens),
        .tokens_per_expert = kittens::py::tensor_to_gl<index_gl>(tokens_per_expert),
        .gate_up_tile_ready = kittens::py::tensor_to_gl<index_gl>(gate_up_tile_ready),
        .hidden_row_block_ready = kittens::py::tensor_to_gl<index_gl>(hidden_row_block_ready),
        .x_routed_ready = kittens::py::tensor_to_gl<index_gl>(x_routed_ready),
        .topk = topk,
        .swiglu_limit = swiglu_limit.value_or(0.0f),
        .num_comm_sms = num_comm_sms,
        .macrobatch_size = macrobatch_size,
        .minibatch_size = minibatch_size
    };

    if (swiglu_limit.has_value())
        kittens::py::launch_kernel<config, globals_recompute_forward_context, recompute_forward_context_kernel<true>>(g);
    else
        kittens::py::launch_kernel<config, globals_recompute_forward_context, recompute_forward_context_kernel<false>>(g);

    return {x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed};
}
