"""The real-GPU bench matrix: one frozen tree per host, every card's speed / VRAM /
quality / inference runs on real weights and data, everything recorded, one report.
`run/` drives the hosts, `plan/` builds the run queues, `measure/` samples the cards,
`report/` summarizes. The drivers live in `scripts/bench/matrix/`; see
docs/performance/bench-matrix.md."""
