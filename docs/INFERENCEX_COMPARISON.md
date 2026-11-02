# InferenceX data on the Compare page

The Compare page treats InferenceX as a separate measured source. It expands the filter options even when the internal Hugging Face ground truth dataset has no rows for a model or GPU. A runtime selection returns measurements from its own source only; internal GT and InferenceX are not automatically overlaid because they use different benchmark clients. AISimulators predictions are requested at the selected source's measured concurrency levels when the model and GPU are supported. Disaggregated InferenceX configurations are not predicted because AISimulators models an aggregated deployment for this workflow; their measured results remain available without a percentage difference.

Serving backend and backend version are shared filters across the two measured sources. For vLLM releases, labels such as the internal dataset's `vLLM-0.24.0` and InferenceX's `v0.24.0` appear as one `v0.24.0` option. The measured runtime configuration retains the source, precision, and internal benchmark release; choosing a serving version does not imply both sources measured the same workload.

When the selected measured backend and version are available in the AISimulators catalog for the selected GPU, the prediction controls use those values and are disabled. Otherwise the controls remain editable so a supported backend and version can be chosen. The page labels that prediction as a different-backend-or-version reference and omits percentage differences from the results table.

The private Hugging Face dataset [`redhat-performance/configiq-inferencex-comparison`](https://huggingface.co/datasets/redhat-performance/configiq-inferencex-comparison) holds `inferencex-8k1k.json`: public single-turn 8,192 input / 1,024 output token results for models in ConfigIQ's approved-model list (`approvedModels` in `public/config.json`; `testedModels` is the transition key) with a verified InferenceX model ID mapping. This is InferenceX's currently active fixed-length scenario; its 1k/1k and 1k/8k scenarios are deprecated. The page shows the selected measurement date and links to its source run.

The page attributes this benchmark data to SemiAnalysis InferenceX and links to the source and selected run. An [InferenceX collaborator confirmed this attribution requirement](https://github.com/SemiAnalysisAI/InferenceX/issues/4655#issuecomment-6507892332) in response to the proposed private storage, public display, and weekly refresh workflow.

The ConfigIQ server reads this file with `INFERENCEX_HF_TOKEN`, which must be a Hugging Face token with read access to that private dataset. Set it in the local or deployment server environment; never expose it through a `NEXT_PUBLIC_` variable or browser request. The server caches a validated snapshot for five minutes and serves the last successful copy if a later HF request fails. If the token is absent or the dataset cannot be loaded at startup, internal GT remains available and the page shows an InferenceX warning. The private dataset controls storage access, but measured results returned by ConfigIQ's comparison API remain visible to users of that page.

Publish a fresh snapshot from the public InferenceX API with one manual command:

```bash
uv run --group tested-models python scripts/publish-inferencex-snapshot.py
```

The collector discovers models from `/api/v1/availability` and `/api/v1/views/options`, intersects them with the verified `testedModels` mappings, then fetches matching display models from `/api/v1/benchmarks`. The publisher validates the results again, confirms the HF dataset is private, and uploads the JSON directly from memory when measurements change. It requires a local HF login with write access to that dataset. No snapshot file is written to the ConfigIQ checkout. The browser reads ConfigIQ's same-origin comparison API; it does not call InferenceX or the private HF dataset directly. A scheduled refresh belongs to the separate automation work.

InferenceX response times are seconds; the snapshot converts TTFT and TPOT to the milliseconds expected by the internal aggregation code. `output_tput_per_gpu` is per physical chip, so aggregate throughput multiplies it by the physical GPU count, adding prefill and decode counts only for disaggregated runs. Disaggregated rows are rejected unless the prefill tensor-parallel size and prefill GPU count are present, so aggregate throughput cannot be undercounted. `tp` remains the decode tensor-parallel size, and `tp_prefill` preserves the prefill size for disaggregated runs. The runtime selector keeps backend, image version, precision, speculation, topology, offload mode, and curve separate so unlike runs are not averaged together.

The approved-model filter limits which models enter the snapshot; it does not guarantee AISimulators supports every selected GPU, serving backend, or version. Models outside the approved list are excluded, even if InferenceX publishes measurements for them. InferenceX and AISimulators may use different request-length distributions, runtime settings, and serving topologies, so the plotted difference is diagnostic rather than a like-for-like accuracy score.

Sources: [InferenceX API reference](https://inferencex.semianalysis.com/api), [InferenceX scenario policy](https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/docs/MODELS.md).
