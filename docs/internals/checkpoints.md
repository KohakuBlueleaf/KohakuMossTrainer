# Checkpoint writes

How a checkpoint reaches the disk, and what it costs in host memory and disk time.

| file | writer | extra host memory at peak |
| --- | --- | --- |
| `trainable.safetensors`, `ema.safetensors`, the EMA snapshots | `kohakumoss.training.checkpoint.stream.save_stream` | one tensor |
| `optim.pt` | `torch.save` | one storage at a time; under FSDP2 the full host state, gathered before the write ([below](#optimizer-state)) |

## Where to write

The write runs at the speed of the target disk. A 5.1 B full finetune (fp32 masters, two
moments, EMA) is tens of GB, so the checkpoint directory belongs on local NVMe. On a
slow or shared disk, write to a fast scratch directory and copy afterwards.

## Streaming safetensors writes

`save_stream` writes in two passes:

1. The safetensors header (dtype, shape and byte offsets of every tensor, in state
   order) is built from metadata alone.
2. Each tensor is then copied to the host and written by itself. CPU contiguous tensors
   are written in place with no copy.

The extra host memory at peak is one tensor; a writer that copies the whole state to
the host first holds a second full copy. `tests/training/checkpoint/test_checkpoint_stream.py`
measures the peak-RSS growth of a fresh process for both writers, and checks that the
safetensors (Rust) reader loads the written file exactly.

## Optimizer state

`optim.pt` goes through `torch.save`, which writes one storage at a time. Under FSDP2 the
parts are gathered to full host tensors on every rank before the write
([multi-gpu.md](../guides/multi-gpu.md)).
