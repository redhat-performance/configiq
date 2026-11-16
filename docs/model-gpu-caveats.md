# Model and GPU caveats

This page records judgement that the catalog cannot express. Catalog endpoints
say what is supported; this page says what to watch out for when the answer is
technically valid but practically risky.

## Precision is hardware-dependent

- **FP8 weights** need hardware with FP8 support. On older devices the service
  rejects the request rather than quietly running a different precision.
- **NVFP4** is a Blackwell-class format. Selecting it on an older GPU is an
  error, not a silent downgrade to FP16.
- **FP8 KV cache** is a separate capability from FP8 weights. A model can fit
  in FP8 while its KV cache cannot, and the two are reported separately.

## Parallelism is not free

Tensor parallelism splits one model copy across GPUs. It makes a model fit, but
every layer now needs an all-reduce across those GPUs. A PCIe-only topology at
high TP can be slower than a smaller configuration that fits on fewer devices,
so a larger TP is not automatically better. Predict reports the result for the
topology you ask for; it does not reward GPU count.

Pipeline parallelism has its own trade-off: it splits layers into stages, and a
bubble appears whenever stages are unevenly loaded.

For Mixture-of-Experts models, the expert-parallel group must match the
attention group. The service resolves the expert parallelism when you do not set
it, including for each disaggregated pool. Do not derive it in a UI or a script:
the constraint belongs to the simulator, and getting it wrong produces a
configuration that cannot be deployed.

## Attention design changes KV growth

Grouped-query, multi-query, multi-head latent, sliding-window, and state-space
models do not all grow the KV cache at the same rate. The service reads the
model configuration and applies the architecture's own rule. Two models with the
same parameter count can therefore have very different concurrency ceilings.

Sliding-window attention is the one most often missed: a long context does not
necessarily cost KV cache in proportion to its length, because old tokens fall
out of the window.

## Quantised checkpoints are not interchangeable

A quantisation label describes a checkpoint that was produced in a particular
way. Two checkpoints with the same nominal precision can have different
calibration, different kernels, and different quality. Prefer checkpoints that
have been validated for the model and backend you are using; treat an arbitrary
community quantisation as unvalidated until it has been run.

## Disaggregated serving needs a fabric

Prefill and decode pools in a disaggregated deployment exchange KV cache. The
interconnect's bandwidth and latency are therefore part of the performance
model. A fabric is a hardware and topology choice, not a simulator setting, and
a simulator that does not model the transfer will say so rather than assume one.

Do not infer a transfer bandwidth from a GPU's inter-node bandwidth. They
describe different links.

## What the estimate does not include

- No multi-tenant isolation, scheduler contention, or noisy-neighbour effects.
- No network, storage, or tokenisation overhead.
- No failure, retry, or autoscaling behaviour.
- No benefit from batching across separate workloads.

Memory is not performance either: a configuration that fits can still miss its
latency target, and a faster configuration can still run out of memory at the
context length you actually use.

## Related

- [Tested models](tested-models.md): how confidence is reported
- [Settings and simulators](settings.md): capabilities come from the service
