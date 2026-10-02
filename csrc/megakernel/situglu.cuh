#pragma once

enum class GluActivation {
    SWIGLU,
    SITUGLU,
};

// beta1 and beta2 are the constants used by situ-glu in the kimi k3 technical report.
static constexpr float SITUGLU_GATE_BETA = 4.0f;
static constexpr float SITUGLU_UP_BETA = 25.0f;

struct situglu_backward_terms {
    float hidden;
    float d_gate;
    float d_up;
};

static __device__ __forceinline__ float situglu_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

static __device__ __forceinline__ float situglu_forward(float gate, float up) {
    // h = [beta1 * tanh(g / beta1) * sigmoid(g)] * [beta2 * tanh(u / beta2)].
    const float gate_factor = SITUGLU_GATE_BETA * tanhf(gate / SITUGLU_GATE_BETA) * situglu_sigmoid(gate);
    const float up_factor = SITUGLU_UP_BETA * tanhf(up / SITUGLU_UP_BETA);
    return gate_factor * up_factor;
}

static __device__ __forceinline__ situglu_backward_terms situglu_backward(float gate, float up) {
    const float gate_tanh = tanhf(gate / SITUGLU_GATE_BETA);
    const float up_tanh = tanhf(up / SITUGLU_UP_BETA);
    const float sigmoid = situglu_sigmoid(gate);
    const float gate_factor = SITUGLU_GATE_BETA * gate_tanh * sigmoid;
    const float up_factor = SITUGLU_UP_BETA * up_tanh;

    // d gate_factor / dg = (1 - tanh(g / beta1)^2) * sigmoid(g)
    //                        + beta1 * tanh(g / beta1) * sigmoid(g) * (1 - sigmoid(g)).
    const float d_gate_factor = (1.0f - gate_tanh * gate_tanh) * sigmoid
                              + gate_factor * (1.0f - sigmoid);
    // d up_factor / du = 1 - tanh(u / beta2)^2.
    const float d_up_factor = 1.0f - up_tanh * up_tanh;
    return {gate_factor * up_factor, d_gate_factor * up_factor, gate_factor * d_up_factor};
}
