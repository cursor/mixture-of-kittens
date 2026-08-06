#include "mok_megakernel.cuh"
#include "mxfp8.cuh"
#include "scheduler.cuh"
#include "utils.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("all_gather_top_experts", &utils::all_gather_top_experts::entrypoint, "",
          pybind11::arg("top_experts"), pybind11::arg("all_gather_top_experts_buffer"),
          pybind11::arg("all_gather_top_experts_buffer_multicast_ptr"), pybind11::arg("rank"), pybind11::arg("chunk_bytes"));
    m.def("barrier_all", &utils::barrier_all::entrypoint, "",
          pybind11::arg("barrier_buffer"), pybind11::arg("barrier_buffer_ptrs"),
          pybind11::arg("barrier_buffer_multicast_ptr"), pybind11::arg("target"));
    m.def("schedule", &scheduler::schedule, "",
          pybind11::arg("topk_all"), pybind11::arg("num_local_experts"), pybind11::arg("schedule_capacity"), pybind11::arg("rank"));
    m.def("mxfp8_quantize", &mxfp8_quantize::mxfp8_quantize_entrypoint, "",
          pybind11::arg("x_bf16"),
          pybind11::arg("return_normal"), pybind11::arg("return_transposed"));
    auto bind_glu_fwd_mxfp8 = [&](const char *name, auto function) {
        m.def(name, function, "",
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("combine_buffer"), pybind11::arg("combine_buffer_ptrs"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_routed_gate"), pybind11::arg("w_routed_gate_sc"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_routed_up"), pybind11::arg("w_routed_up_sc"),
          pybind11::arg("w_shared_down"), pybind11::arg("w_routed_down"), pybind11::arg("w_routed_down_sc"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("num_comm_sms"),
          pybind11::arg("macrobatch_size"), pybind11::arg("minibatch_size"));
    };
    bind_glu_fwd_mxfp8("dispatch_mlp_swiglu_combine_fwd_mxfp8",
                       &dispatch_mlp_glu_combine_fwd_mxfp8<GluActivation::SWIGLU>);
    bind_glu_fwd_mxfp8("dispatch_mlp_situglu_combine_fwd_mxfp8",
                       &dispatch_mlp_glu_combine_fwd_mxfp8<GluActivation::SITUGLU>);

    auto bind_glu_bwd_mxfp8 = [&](const char *name, auto function) {
        m.def(name, function, "",
          pybind11::arg("d_y_buffer"), pybind11::arg("d_y_buffer_ptrs"),
          pybind11::arg("d_x_routed_buffer"), pybind11::arg("d_x_routed_buffer_ptrs"),
          pybind11::arg("router_weight_buffer"), pybind11::arg("router_weight_buffer_ptrs"),
          pybind11::arg("d_router_weight_buffer"), pybind11::arg("d_router_weight_buffer_ptrs"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_routed_gate_T"), pybind11::arg("w_routed_gate_T_sc"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_routed_up_T"), pybind11::arg("w_routed_up_T_sc"),
          pybind11::arg("w_shared_down"), pybind11::arg("w_routed_down_T"), pybind11::arg("w_routed_down_T_sc"),
          pybind11::arg("x_fp8_t_routed"), pybind11::arg("x_sc_t_routed"),
          pybind11::arg("gate_shared"), pybind11::arg("gate_fp8_routed"), pybind11::arg("gate_sc_routed"),
          pybind11::arg("up_shared"), pybind11::arg("up_fp8_routed"), pybind11::arg("up_sc_routed"),
          pybind11::arg("hidden_shared"), pybind11::arg("hidden_fp8_t_routed"), pybind11::arg("hidden_sc_t_routed"),
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("w_routed_gate"), pybind11::arg("w_routed_gate_sc"),
          pybind11::arg("w_routed_up"), pybind11::arg("w_routed_up_sc"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("num_comm_sms"),
          pybind11::arg("macrobatch_size"), pybind11::arg("minibatch_size"));
    };
    bind_glu_bwd_mxfp8("dispatch_mlp_swiglu_combine_bwd_mxfp8",
                       &dispatch_mlp_glu_combine_bwd_mxfp8<GluActivation::SWIGLU>);
    bind_glu_bwd_mxfp8("dispatch_mlp_situglu_combine_bwd_mxfp8",
                       &dispatch_mlp_glu_combine_bwd_mxfp8<GluActivation::SITUGLU>);

    auto bind_glu_fwd_bf16 = [&](const char *name, auto function) {
        m.def(name, function, "",
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("combine_buffer"), pybind11::arg("combine_buffer_ptrs"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_routed_gate"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_routed_up"),
          pybind11::arg("w_shared_down"), pybind11::arg("w_routed_down"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("num_comm_sms"),
          pybind11::arg("macrobatch_size"), pybind11::arg("minibatch_size"));
    };
    bind_glu_fwd_bf16("dispatch_mlp_swiglu_combine_fwd_bf16",
                      &dispatch_mlp_glu_combine_fwd_bf16<GluActivation::SWIGLU>);
    bind_glu_fwd_bf16("dispatch_mlp_situglu_combine_fwd_bf16",
                      &dispatch_mlp_glu_combine_fwd_bf16<GluActivation::SITUGLU>);

    auto bind_glu_bwd_bf16 = [&](const char *name, auto function) {
        m.def(name, function, "",
          pybind11::arg("d_y_buffer"), pybind11::arg("d_y_buffer_ptrs"),
          pybind11::arg("d_x_routed_buffer"), pybind11::arg("d_x_routed_buffer_ptrs"),
          pybind11::arg("router_weight_buffer"), pybind11::arg("router_weight_buffer_ptrs"),
          pybind11::arg("d_router_weight_buffer"), pybind11::arg("d_router_weight_buffer_ptrs"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_routed_gate"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_routed_up"),
          pybind11::arg("w_shared_down"), pybind11::arg("w_routed_down"),
          pybind11::arg("x_routed"),
          pybind11::arg("gate_shared"), pybind11::arg("gate_routed"),
          pybind11::arg("up_shared"), pybind11::arg("up_routed"),
          pybind11::arg("hidden_shared"), pybind11::arg("hidden_routed"),
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("num_comm_sms"),
          pybind11::arg("macrobatch_size"), pybind11::arg("minibatch_size"));
    };
    bind_glu_bwd_bf16("dispatch_mlp_swiglu_combine_bwd_bf16",
                      &dispatch_mlp_glu_combine_bwd_bf16<GluActivation::SWIGLU>);
    bind_glu_bwd_bf16("dispatch_mlp_situglu_combine_bwd_bf16",
                      &dispatch_mlp_glu_combine_bwd_bf16<GluActivation::SITUGLU>);
    m.def("fwd_epilogue", &utils::fwd_epilogue, "",
          pybind11::arg("y_shared"), pybind11::arg("combine_buffer"), pybind11::arg("topk_weights"));
    m.def("bwd_epilogue", &utils::bwd_epilogue, "",
          pybind11::arg("d_x_shared"), pybind11::arg("d_x_routed_buffer"));
}
