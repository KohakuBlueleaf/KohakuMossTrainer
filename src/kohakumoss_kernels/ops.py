"""Op names and their eager signatures — the contract every backend implements.

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

Optional fused ops (dense base, no adapter; the composition of the primitives is the spec):

    linear_qk_rope(y, wq, wk, q_weight, k_weight, pos, bands, eps, heads) -> (q, k)
    linear_situ(y, w_up_gate, bias) -> situ_glu(y W^T + b)
    linear_gated_residual(a, w, bias, x, gate, seg, r) -> gated_residual(x, a W^T + b, ...)

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
ALL = NORM + ROPE + ATTENTION + FFN + RESIDUAL + LOSS + FUSED
