"""Op names and their eager signatures, implemented by every backend.

Segment ids `seg` are nondecreasing (sample-major packing); segment reductions rely on it.

    rms_norm(x[N,D], weight[D] | None, eps) -> y[N,D]
    rms_norm_mod(x[N,D], weight[D] | None, shift[B,D] | None, scale[B,D] | None,
                 seg[N] int32 | None, eps) -> y[N,D]
    qk_norm_rope(q[N,H,Dh], k[N,H,Dh], q_weight[Dh], k_weight[Dh], pos[N,A] f32,
                 bands[Dh//(2A)] f32, eps) -> (q[N,H,Dh], k[N,H,Dh])
    attention_varlen(q[N,H,Dh], k[N,H,Dh], v[N,H,Dh], cu_seqlens[B+1] int32,
                     max_seqlen: int, scale: float) -> o[N,H,Dh]
    situ_glu(x[N,2F]) -> a[N,F]
    gated_residual(x[N,D], o[N,D], gate[B,D] | None, seg[N] int32 | None, r: float)
        -> x*r + o*gate[seg]
    token_mse(pred[N,C], target[N,C], seg[N] int32, weight[B] f32, mask[N] f32 | None)
        -> f32 scalar sum of weight[seg]*mask*mean_c((pred-target)^2)

Optional fused ops (dense base, no adapter; the composition of the primitives is the
eager reference):

    linear_qk_rope(y, wq, wk, q_weight, k_weight, pos, bands, eps, heads) -> (q, k)
    linear_situ(y, w_up_gate, bias) -> situ_glu(y W^T + b)
    linear_gated_residual(a, w, bias, x, gate, seg, r) -> gated_residual(x, a W^T + b, ...)

Rotation and the rotated quantized linear (ConvRot weights; H_g the normalised regular
Hadamard of `kohakumoss_kernels.hadamard`, R = blockdiag(H_g) over the last dim):

    group_hadamard(x[..., K], group: int) -> x R [..., K]   (self-adjoint, involutory)
    quant_linear_rotated(x[N,K], q[M,K], scale[M], bias[M] | None, fmt, group: int)
        -> quant_linear(x R, q, scale, bias, fmt); dx = (dy (q * scale)) R
    int8_linear(x[N,K], q[M,K] int8, scale[M], bias[M] | None, group: int, dgrad)
        -> Q(x R) q^T * s_x * scale + bias (int8 GEMM, int32 accumulation; group 0: no
        R); dx = Q(dy * scale) q * s_dy R (dgrad "int8") or (dy (q * scale)) R ("16bit")

Text-encoder ops (Qwen3.5 Gated DeltaNet + causal GQA attention; forward only):

    causal_conv_varlen(x[N,C], weight[C,W], cu_seqlens[B+1] int32)
        -> silu(depthwise causal conv of width W, left padding reset per segment)[N,C]
    gated_delta_rule_varlen(q[N,Hk,Dk], k[N,Hk,Dk], v[N,Hv,Dv], a[N,Hv], b[N,Hv],
                            a_log[Hv], dt_bias[Hv], cu_seqlens[B+1] int32,
                            max_seqlen: int) -> o[N,Hv,Dv]
        g = -exp(a_log) * softplus(a + dt_bias), beta = sigmoid(b), q and k l2-normalised
        (q also by Dk^-1/2), value head h reads key head h // (Hv/Hk);
        S_t = exp(g_t) S_{t-1} + beta_t k_t (v_t - exp(g_t) S_{t-1}^T k_t)^T,
        o_t = S_t^T q_t, S reset at every segment start.
    rms_norm_gated(x[N,D], z[N,D], weight[D], eps) -> x / rms(x) * weight * silu(z)
    attention_varlen_causal(q[N,H,Dh], k[N,Hkv,Dh], v[N,Hkv,Dh], cu_seqlens[B+1] int32,
                            max_seqlen: int, scale: float) -> o[N,H,Dh]
        causal inside segments; query head h reads kv head h // (H/Hkv).

Adapter math and kernels come from the lycoris-lora package. Shapes are flat
token-major; outputs are in the input activation dtype; reductions are >= fp32.
"""

NORM = ("rms_norm", "rms_norm_mod")
ROPE = ("qk_norm_rope",)
ATTENTION = ("attention_varlen",)
FFN = ("situ_glu",)
RESIDUAL = ("gated_residual",)
LOSS = ("token_mse",)
FUSED = ("linear_qk_rope", "linear_situ", "linear_gated_residual")
OPTIM = ("adamw_step", "adamw8bit_step", "adamw_step_multi")
QUANT = (
    "quant_linear",
    "quant_linear_rotated",
    "int8_linear",
    "fp8_linear",
    "fp8_linear_infer",
    "fp8_linear_train",
)
ROTATION = ("group_hadamard",)
TEXT = (
    "causal_conv_varlen",
    "gated_delta_rule_varlen",
    "rms_norm_gated",
    "attention_varlen_causal",
)
ALL = (
    NORM + ROPE + ATTENTION + FFN + RESIDUAL + LOSS + FUSED + OPTIM + QUANT + ROTATION
    + TEXT
)  # fmt: skip
