# Tested-model automation

This directory is the replacement home for the tested-model research pipeline.
It will generate datasets, leakage-safe train/validation/test splits, Triton
classifier artifacts, registry metadata, and dashboard payloads.

The end-to-end data contract and host deployment procedure are documented in
[`docs/tested-models.md`](../../docs/tested-models.md). This README focuses on
running the scripts and tests.

## Setup

From the repository root, install the shared Python automation environment with
uv:

```bash
uv sync --group tested-models
```

The group is defined in the root `pyproject.toml`. The local
`requirements.txt` is retained for CI or environments that use pip instead of
uv.

Set the source endpoint outside the repository when running extraction:

```bash
export PERF_DATA_API_URL="https://<approved-data-source>"
uv run --group tested-models python scripts/tested_models/discover_data.py --accelerator H200
```

The endpoint is never written to discovery or dataset manifests. Future source
adapters can use the same pair and parquet contracts without exposing their
connection details.

The generated parquet files are local build inputs and are git-ignored. The
extractor writes only classification-relevant identity, workload, serving
configuration, precision, prefix, reliability, throughput, and latency
columns. Run IDs, UUIDs, MLflow IDs, timestamps, raw runtime arguments, and
other administrative fields are dropped before parquet is written.

`legacy_mvp.py` is retained temporarily as a migration reference from the
former `configiq-tested-models` repository. It must not be used to publish
production artifacts because it contains placeholder knee labeling, row-level
validation, and Python pickle outputs.

The production pipeline will be added incrementally in this directory:

```text
discover_data.py       # Find available model/hardware ground truth
extract_datasets.py    # Write normalized parquet datasets
split_dataset.py       # Create immutable grouped data splits
label_data.py          # Generate labels and knee metadata
train_classifiers.py  # Train and evaluate Triton-compatible models
export_triton.py       # Export validated classifier artifacts
validate_artifacts.py  # Verify bundle and schema integrity
build_registry.py      # Generate the tested-model registry
build_dashboard.py     # Generate dashboard data
```

The scripts are intentionally kept independent of the ConfigIQ web runtime.
Internal extraction is run only from an approved environment; GitHub Actions
consumes the resulting pinned public Hugging Face dataset revision.

The repeatable GitHub workflow is `.github/workflows/tested-models.yml`. It
downloads a pinned revision of the public
`redhat-performance/configiq-performance-data` Hugging Face dataset, generates
all artifacts, and uploads only the sanitized registry/Triton/dashboard bundle.
Raw parquet, split data, source identifiers, and training work files are not
uploaded by the workflow.

To publish a new sanitized ground-truth revision after running extraction from
the approved source environment:

```bash
uv run --group tested-models python scripts/tested_models/publish_ground_truth.py
```

Set the GitHub repository variable `TESTED_DATASET_REVISION` to the resulting
Hugging Face commit before starting the automation workflow.

## Stage 5: performance envelope classifiers

Run the complete leakage-safe pipeline from the repository root after
`download_ground_truth.py` has populated `data/tested-models`:

```bash
uv run --group tested-models python scripts/tested_models/split_dataset.py
uv run --group tested-models python scripts/tested_models/label_data.py
uv run --group tested-models python scripts/tested_models/train_classifiers.py
uv run --group tested-models python scripts/tested_models/export_triton.py
uv run --group tested-models python scripts/tested_models/build_dashboard.py
uv run --group tested-models python scripts/tested_models/build_registry.py
uv run --group tested-models python scripts/tested_models/validate_artifacts.py
```

Outputs are written below `data/tested-models/stage5/` and
`data/tested-models/stage6/registry/` (and are ignored by Git): grouped parquet
splits, labeled rows, XGBoost JSON models, metadata, a complete Triton bundle,
and the generated registry. The split seed is explicit and configuration
groups are assigned as a unit, so repeated measurements cannot cross splits.
Label thresholds and statistics are fitted from training rows only. Model
selection uses validation rows; the test metrics are computed once after
selection. Pairs with a single training class are reported and skipped.

The feature schema is recorded in each metadata file, including categorical
vocabularies, numeric medians, feature order, and the 0.50/0.85 decision
thresholds. Only configuration and derived memory-pressure fields are features;
throughput, latency, identifiers, runtime arguments, MLflow fields, and other
measurements are excluded from classifier inputs. The internal source API may
contain operational identifiers and runtime strings; extraction drops them
before writing the sanitized parquet files published to Hugging Face.
