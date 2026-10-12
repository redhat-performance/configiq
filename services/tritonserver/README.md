# ConfigIQ Triton Server

This image combines the standard NVIDIA Triton `26.09-py3` image with:

- Triton's built-in FIL backend for tested XGBoost classifiers.
- vLLM `0.30.0`.
- The Triton vLLM backend pinned by `VLLM_BACKEND_COMMIT` in the `Containerfile`.

The standard image is intentional: the specialized `26.09-vllm-python-py3`
image contains vLLM but does not contain FIL.

Build from the repository root:

```bash
podman build -f services/tritonserver/Containerfile \
  -t configiq-tritonserver services/tritonserver
```

The build verifies that FIL, the vLLM backend, vLLM `0.30.0`, and native
`K2HorizonForCausalLM` support are present. Runtime validation must also run on
an NVIDIA L4 host before publishing the image digest for deployment.
