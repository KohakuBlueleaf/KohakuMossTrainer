## Summary

<!-- What does this PR do? 1-3 sentences. -->

## PR Type

<!-- Check all that apply. -->

- [ ] Bug fix
- [ ] Documentation
- [ ] Tests only
- [ ] Refactor / maintenance
- [ ] Feature / behavior change
- [ ] New kernel or backend
- [ ] Breaking change

## Motivation / Context

<!-- Why is this change needed? What problem does it solve? For a fix: the measured
problem (the log line, the number, the traceback). -->

## Changes

<!-- Bullet points of what changed. -->

-

## Validation

<!-- How did you verify this works? Include the exact commands you ran and on which GPU. -->

```bash
ruff check src scripts tests configs
black --check src scripts tests configs
pytest -q                     # the model-free suite
pytest -q -m real_weights     # if you touched a path the real model runs (needs KOHAKUMOSS_WEIGHTS)
```

## Checklist

- [ ] I read `CONTRIBUTING.md` and `CLAUDE.md` (the code conventions)
- [ ] I linked the relevant issue(s) below
- [ ] If this is a feature / behavior / architecture change, an issue was opened before this PR and a maintainer approved it
- [ ] I kept this PR focused and did not include unrelated changes
- [ ] Code files carry only *what*; every *why* is in `docs/`
- [ ] No imports inside functions; ASCII-only source and messages
- [ ] Every new check has a negative case (`tests/_util/negative.py`) and no fake / stand-in model
- [ ] A new or changed kernel matches the eager reference within the precision bar (fp16 and bf16, forward and backward) and is selected only where it is faster
- [ ] Docs are updated for user-visible behavior, knobs, CLI or config changes
- [ ] `ruff` and `black` pass; the model-free suite passes locally, or I explained below why not
- [ ] Windows still works for any single-GPU feature, or the reason is written in `docs/guides/install-windows.md`

## Related Issues

<!-- Required for feature PRs. Examples:
- Fixes #123
- Approved in #456
-->

## Notes for Reviewers

<!-- Optional: risky areas, follow-ups, tradeoffs. -->

## Logs / Benchmarks (if applicable)

<!-- Speed / VRAM tables (card, profile, tokens, samples/s, peak), stack traces, etc. -->

## Exceptions / Not Run

<!-- If you skipped any checklist item, explain exactly why. -->
