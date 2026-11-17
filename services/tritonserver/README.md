# ConfigIQ Triton Server

This image combines the standard NVIDIA Triton `26.09-py3` image with:

- Triton's built-in FIL backend for tested XGBoost classifiers.
- vLLM `0.30.0` in `/opt/vllm-site`, which the assistant's model server runs
  directly from the same pinned image.

The standard image is intentional: the specialized `26.09-vllm-python-py3`
image contains vLLM but does not contain FIL.

Triton itself serves FIL only. The assistant is a separate container: Triton's
`/v2` API cannot answer a non-streaming request for a decoupled backend, and its
OpenAI frontend does not vendor the tool parsers the assistant needs. The image
therefore no longer contains the Triton vLLM backend or its metrics patch.

Build from the repository root:

```bash
podman build -f services/tritonserver/Containerfile \
  -t configiq-tritonserver services/tritonserver
```

The build verifies that FIL, vLLM `0.30.0` with its `reasoning_parser` option,
and native `K2HorizonForCausalLM` support are present. Runtime validation must
also run on an NVIDIA L4 host before publishing the image digest for deployment.
