# Encode feed

`ENCODE_DEVICES` gives the text encoder and the VAE cards of their own: `DEVICES` only
train, `ENCODE_DEVICES` only encode. The run splits into three kinds of process:

| process | runs | module |
| --- | --- | --- |
| main | the data loaders of every data rank, the encode feed, the batch streams | `training.encode.feed` |
| encoder (one per `ENCODE_DEVICES` card) | the text encoder and the VAE | `training.encode.service` |
| training (one per `DEVICES` card for DDP / FSDP2 / pipeline, else one) | the DiT step only | `training.encode.streamed` |

```python
DEVICES = (0, 1, 2, 3)      # 4x RTX 3090: training
ENCODE_DEVICES = (4, 5)     # 2x RTX 2080 Ti: text encoder + VAE
ENCODE_DEPTH = 32           # micro-batches kept encoded ahead per data stream
ENCODE_CHUNK = 8            # micro-batches per encoder request
```

Every layout works under the feed; a single-process layout (`none`, `threaded`) trains in
one child process of the main process.

## Data flow

1. The main process builds one loader per data rank (it writes the catalog; the training
   processes read it) and one validation loader per rank when `VALIDATE_EVERY` > 0.
2. One producer thread per data stream loads batches as far as `ENCODE_DEPTH` ahead of
   what its training ranks have taken.
3. The dispatcher sends chunks of up to `ENCODE_CHUNK` pending batches (from every
   stream) to one request queue that every encoder process takes from, so the encoder
   cards share the work whatever the rank count. A chunk is gathered when a request slot
   frees (two per encoder process), so it holds everything that arrived meanwhile.
4. An encoder process encodes a chunk in one text-encoder call (prompts cut to their
   `text_limit` tokens before the pass, packed up to `TEXT_ENCODE['max_tokens']` per
   kernel call) and one VAE call (images sorted by size, same sizes batched).
5. The main process finishes each batch as a cache hit carries it: text rounded as an
   online cache stores it and cut to `text_limit`, latents rounded and gathered to the
   kept tokens; an online cache gets the fresh encodings here; the pixels are dropped.
6. Batches go to their training ranks in loader order (a release buffer reorders the
   chunks the encoder processes finish); a layout with one data rank for several ranks
   gets each batch on every rank's queue.

A training process takes ready batches from its queue and moves them to its card; it
never talks to an encoder process. It holds no text encoder; the sampling rank holds a
VAE for decoding samples and gets the sample-prompt embeddings from the feed at build.

## State and resume

A streamed batch carries the loader state after it; the training process's loader state
is the state of the last batch it took, so a checkpoint resumes at the first batch not
trained on. On resume the training process asks the feed to seek: once every rank of the
stream asked, the stream restarts there under a new generation, and batches of the old
one are dropped on both sides.

## Failure

A failing encoder process, loader or feed thread logs its traceback in the main process
and sets the feed's failure event; every training process raises `FeedError` at its next
wait and the launcher ends the run. An encoder process ends when the main process is
gone.

## Metrics

- `feed_wait`: seconds per step the training process waited for a ready batch; 0 while
  the feed keeps ahead.
- `feed_encode`: encoder-process seconds per step its batches took (text encoder + VAE,
  each chunk's time shared by tokens / images). Below the step time times the number of
  ranks per encoder card, the encoder cards keep up.
- `te_*` / `vae_*`: the text-encoder and VAE shares, as for in-process encoders.

## Process setup

An encoder process sets up logging, the allocator, the compile pool and TUNE as a
training process does, makes its card current and prewarms its text encoder before the
first pass. Tensors cross the queues through shared memory (`file_system` sharing).
