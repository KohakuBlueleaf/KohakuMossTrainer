# Checkpoint writes

`trainable.safetensors`, `ema.safetensors` and the EMA snapshots are written by
`kohakumoss.training.checkpoint.stream.save_stream`:

1. The safetensors header (dtype, shape and byte offsets of every tensor, in state
   order) is built from metadata alone.
2. Each tensor is then copied to the host and written by itself. CPU contiguous tensors
   are written in place with no copy.

The extra host memory at peak is one tensor. A writer that copies the whole state to
the host first holds a second full copy. tests/training/checkpoint/test_checkpoint_stream.py
measures the peak-RSS growth of a fresh process for both writers. It also checks that the
safetensors (Rust) reader loads the written file exactly.

`optim.pt` goes through `torch.save`, which writes one storage at a time. Under FSDP2 the
parts are gathered to full host tensors on every rank before the write
([multi-gpu.md](../guides/multi-gpu.md)).

The write runs at the speed of the target disk. A 5.1 B full finetune (fp32 masters, two
moments, EMA) is tens of GB, so the checkpoint directory belongs on local NVMe. On a
slow or shared disk, write to a fast scratch directory and copy afterwards.
