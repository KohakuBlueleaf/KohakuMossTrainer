# Healing the stream and the weights during a finetune

MOSS-Image base carries two related problems: a residual stream whose energy sits in a
few massive channels ([massive-channels.md](massive-channels.md)), and sparse weight
elements far above their tensor's RMS. This page shows what a penalty on the stream
does to them in a real finetune, why it leaves the weights alone, and the two pieces
KohakuMoss adds for a finetune that heals both, gradually:

- the `write` term of `STREAM_LOSS`: the concentration of what each sublayer writes;
- `OUTLIER_DECAY`: a decoupled decay of each weight element's excess over a threshold.

The reference for every knob is [training.md](training.md#stream-penalties). All
real-model figures are measured on 6 text-image samples at 1024² area, bf16
forward, fp32 statistics. "Finetuned" is a full finetune of the base for 10k steps at
lr 2e-5 carrying the state concentration term (weight 0.01, `k = 32`) and the `rms` term
(2e-5). `C` is the set of the 32 highest-energy channels of the stream entering a block.

## 1. What a stream penalty does in a finetune

| | base | finetuned |
| --- | --- | --- |
| largest \|x\| in the stream | 59.3k | 7.1k |
| mean of the top 1e-4 of \|x\|, worst block | 1604 | 585 |
| stream RMS, mean over blocks | 27.8 | 9.9 |
| top-32 energy share, mean over blocks | 0.84 | 0.51 |
| top-32 share after the finetune, blocks 0–6 / blocks 20–40 | | 0.95–0.97 / 0.08–0.49 |
| velocity cosine to base | 1 | 0.9974 |
| flow loss | 0.4277 | 0.4315 |

The stream looks healthier and the model keeps its outputs. The way it gets there is
not the way the penalty intended:

- **Weights: unchanged.** Every projection's max |w| / RMS, top-1e-4 / RMS, kurtosis and
  row / column energy shares are the same to three digits (`layers.0` `ffn_down`: max
  103 × RMS, kurtosis 13.5, before and after). The overall weight change is 1.7 %
  relative, concentrated in the conditioning (time embedding 27 %, head modulation 14 %).
- **Magnitude: through the conditioning.** Swapping only the 8 conditioning tensors
  (time embedding, shared block modulation, head modulation) from the finetuned model
  into the base carries about half of the magnitude drop (largest value 48.9k → 22.0k,
  RMS −45 %) and almost none of the concentration change. Block 0's FFN write peak falls
  16.6k → 3.9k with block 0's own weights unchanged: the modulation turns it down.
- **Concentration: by cancellation.** In blocks 14–39 the FFN writes point *against* the
  inherited massive channels and the attention writes point *along* them. The share of
  the inherited `C` mass that an FFN write removes rises from 7–11 % (base) to 45–102 %
  per block (finetuned), and the attention write of the same block adds about as much
  back. The base model has the same pattern at about 1/5 the size. Both writes keep
  about 0.9 of their energy inside `C`.

A penalty on the block *outputs* is met just as well by a later block writing the
negative of an earlier block's massive channel. Each writer stays as concentrated as
before; only their sum is spread out. The outliers that create the channels stay in the
weights, and a later finetune inherits them.

## 2. Why a loss does not reach the weight outliers

Elements above a multiple of their tensor's RMS, joint blocks of the base model (about
5e9 elements):

| threshold | elements |
| --- | --- |
| 32 × RMS | ~120 (60 in `ffn_down`, 45 in `ffn_up_gate`, 8 in `k`, 6 in `q`, 1 in `o`) |
| 16 × RMS | ~2,400 |
| 8 × RMS | ~72,000 |

The largest: `layers.41` `ffn_up_gate` 113 ×, `layers.0` `ffn_down` 103 ×, `layers.41`
`k` 63 ×; outside the blocks, the image embedder at 20.6 × (its 32 largest rows hold
26.6 % of its row energy).

Two reasons the stream penalty and the task leave them in place:

- **The stream gradient does not point at them.** The attribution `|w · g|` of every
  stream term on the elements above 20 × RMS is at most 2.3 % of its total in every
  block group (12 % in the conditioning for the state term).
- **Adam bounds the move.** Adam's normalised update moves an element by about `lr` per
  step whatever the size of its gradient. At lr 2e-5 over 10k steps that is 0.2 at
  most, against an element of 5.1 (`layers.0` `ffn_down`: 103 × an RMS of 0.049). A
  penalty on the weights added to the loss has the same limit (toy below: exactly
  `lr × steps`).

## 3. A toy with a planted massive channel

A pre-norm residual model (width 64, 8 blocks) trained on a regression task, with one
writer element planted at 40 × its tensor's RMS; the uniform top-4 share is 0.0625.
Each row is one penalty configuration trained with Adam for the same steps.

| configuration | task loss | outlier / RMS | state share | channel size, block 1 / 8 | cancellation |
| --- | --- | --- | --- | --- | --- |
| task only | 0.0034 | 40.2 | 0.32–0.50 | 5.1 / 7.5 | ~0 |
| state concentration | 0.0037 | 38.6 | 0.066 | 1.0 / 1.4 | block 0 writes against it |
| write concentration, full denominator | 0.0037 | 37.2 | 0.08–0.27 | 1.1 / 3.2 | adds |
| write concentration, detached denominator | 0.0060 | 36.5 | 0.09 | 0.41 / 0.64 | ~0 |
| weight penalty in the loss | 0.0035 | 18.8 (the `lr × steps` bound) | 0.20–0.46 | 2.9 / 5.6 | adds |
| decoupled weight decay alone | 0.0035 | 4.7 | 0.18–0.46 | 2.6 / 5.4 | adds |
| detached write + decoupled decay | 0.0065 | 4.7 | 0.09 | 0.30 / 0.53 | ~0 |

- The state term is met by cancellation, as in the real finetune.
- A write share with the full total in its denominator is met by **dilution**: every
  write grows all its other channels until its share is low, while still adding the
  massive channel.
- With the total detached (shrink-only), neither dodge works: the channel falls 8–12 ×.
- The decoupled decay alone removes the outlier, and the model rebuilds the channel
  from other elements.
- Both together leave the weights and the stream healthy. The task cost is that of a
  channel the task was using.

## 4. The write term

For each joint block and each sublayer, the write is what the sublayer adds to the
stream: `out − x · r` for attention (`r` its residual scale) and `out − x` for the FFN.
With `e` the write's per-channel energy over the micro-batch's tokens:

    c = topk_k(e) / sg(sum(e))          share of the write in its k largest channels
    loss = weight · mean over (block, sublayer) of relu(c − τ(step))²

- **Each writer, not the sum.** A block that cancels an inherited channel writes a
  concentrated (negative) write and is priced for it, the same as the block that
  created the channel.
- **Shrink-only (`dilute=False`, the default).** `sg` detaches the total, so the only
  way down is to shrink the top channels; growing every other channel is no longer a
  way to lower the share.
- **Scale-free.** Scaling a whole write (a gate, the modulation) does not change `c`;
  the gradient moves the write's channel mix. The stream's level is the `rms` term's
  job.
- **A target, because base writes are concentrated.** In the base model the top-32
  share is 0.83–0.95 for attention writes and 0.78–0.90 for FFN writes in blocks 1–35,
  about 0.99 in block 0, nearly flat over σ. A write with no concentration at all is
  not a working write, so a plain penalty on `c` would pull against the task forever.
  The hinge stops at `target`: a write at or below it adds no loss and no gradient.
- **An anneal, because the heal starts from the base.** `τ` is 1 until `start` (loss
  and gradient zero) and falls linearly to `target` over `anneal` steps. With
  `target = 0.5` and `anneal = 2000`, `τ` passes 0.95 at step 200, 0.83 at step 680 and
  0.78 at step 880: the most concentrated writes are engaged first, each one from zero
  pressure, and no step sees a sudden push.
- **Gradient size.** At weight 1, the shrink-only term's gradient norm is 22–25 × the
  task loss's on the base (attention / FFN part) and 31–39 × on the finetuned model;
  its cosine with the task gradient is −0.04 to −0.09. A weight near 0.004 puts it at
  about a tenth of the task gradient. The state term, for comparison: 1.4 × on the base
  and 6.5 × at 10k steps.
- **Cost.** One fp32 channel-energy vector per sublayer per block; autograd keeps the
  write for the norm's backward, which a block checkpoint recomputes instead.

## 5. Outlier decay

After the optimiser step, on every targeted weight (every trained 2-D weight by
default: block projections, embedders, conditioning, final linear):

    p -= lr_t · λ(t) · sign(p) · relu(|p| − c(t) · r0)

`r0` is the tensor's RMS at the start of the run, fixed for the run.

- **Decoupled, like AdamW's decay.** It is the gradient of `λ/2 · relu(|p| − c r0)²`,
  applied outside Adam, so the move is proportional to the excess instead of the
  `lr`-sized sign step Adam would give it. An excess shrinks with a time constant of
  `1 / (λ · lr)` steps and follows the lr schedule.
- **No clamp and no direct write.** No element is ever set to a value. At λ 6 and
  lr 2e-5 an element loses 1.2e-4 of its excess per step (a time constant of about
  8.3k steps). The move is zero at the threshold and grows continuously with the
  excess, so crossing the threshold changes nothing suddenly.
- **Why a threshold `c`.** It is what makes this an outlier decay. With `c = 0` the
  formula is plain decay of every element, `p -= lr·λ·p`: it shrinks the whole model
  and leaves the concentration where it was. Measured in units of the tensor's own
  start RMS, the threshold is scale-free across tensors. A fixed `r0` cannot be met by
  growing every other element, which a reference that followed the live RMS would
  allow.
- **Against the task.** Where the task holds an element large, Adam pushes it back by
  about `lr` per step and the decay pulls by `lr · λ · excess`. The element settles at
  an excess of about `1 / λ` (absolute) over the threshold, not at the threshold: the
  task keeps what it needs.
- **Gradual.** `c(t)` falls geometrically from `cap_start` to `cap` over `cap_anneal`
  steps, so the largest elements are engaged first (above 64 × RMS: fewer than the
  ~120 above 32 ×). `λ` ramps linearly from 0 over `warmup` steps.
- **Every tensor, so also the hidden state inside the blocks.** The decay covers
  `ffn_up_gate` and `q` / `k` as well as the writers: the outliers that make large FFN
  hidden units or attention logits are treated the same as those that write the
  stream.
- **Cost.** One elementwise pass over the targeted tensors per step. Full finetune with
  fp32 device-stepped weights (a step of `lr · λ` of an element is below a 16-bit
  weight's resolution).

## 6. The combination

| piece | heals | covers |
| --- | --- | --- |
| `write` | the stream's concentration, at every writer, without cancellation or dilution | the readers' inputs |
| `rms` | the stream's level | the readers' inputs, fp16 headroom |
| `OUTLIER_DECAY` | the weight outliers, including those behind the in-block hidden state | the weights a later finetune inherits |

The toy shows why both the stream and the weight side are needed. The decay alone lets
the model rebuild the channel from other elements; the write term alone heals the
stream and leaves the outliers in place.

A configuration for a long full finetune at lr 2e-5:

```python
STREAM_LOSS = {
    "write": {"weight": 5e-3, "target": 0.5, "anneal": 2000},
    "rms": {"weight": 5e-5},
}
OUTLIER_DECAY = {
    "rate": 6.0,           # time constant ~8.3k steps at lr 2e-5; ~12 at lr 1e-5
    "cap": 16.0,
    "cap_start": 64.0,
    "cap_anneal": 10000,
    "warmup": 1000,
}
```

- `opt/outlier_over` at step 1 shows how many elements the decay engages. If the
  conditioning tensors dominate, `exclude=r"^(time_embedding|shared_block_modulation|video_head_modulation)\."`
  leaves them to the task.
- `stream/write_attn` and `stream/write_ffn` should follow `stream/write_target` down
  and settle near it; `opt/outlier_max` should fall slowly, on the decay's time
  constant.

## Open points

- A full run with this configuration is pending: its weight statistics, write shares,
  stream statistics and image quality against the finetune of section 1.
- Which FFN hidden units produce the massive channels, and how the shared modulation's
  per-channel gate and scale act on `C` at each σ, are not measured yet.
