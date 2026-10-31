"use client";

import React, { useCallback, useRef, useState } from "react";
import {
  ActionList,
  ActionListItem,
  Button,
  Card,
  CardBody,
  CardTitle,
  DescriptionList,
  DescriptionListDescription,
  DescriptionListGroup,
  DescriptionListTerm,
  EmptyState,
  EmptyStateBody,
  FormGroup,
  FormHelperText,
  HelperText,
  HelperTextItem,
  Label,
  NumberInput,
  Spinner,
  Split,
  SplitItem,
  TextInput,
  Title,
} from "@patternfly/react-core";
import {
  Chart,
  ChartAxis,
  ChartScatter,
  ChartVoronoiContainer,
  ChartLine,
} from "@patternfly/react-charts/victory";
import DownloadIcon from "@patternfly/react-icons/dist/esm/icons/download-icon";
import CheckCircleIcon from "@patternfly/react-icons/dist/esm/icons/check-circle-icon";
import ExclamationTriangleIcon from "@patternfly/react-icons/dist/esm/icons/exclamation-triangle-icon";

import type { Candidate, CandidateViews, ParetoRequest } from "@/lib/types/pareto";
import { computeFrontier, pickRecommendation } from "@/lib/pareto/frontier";
import { streamCandidates } from "@/lib/api/pareto-stream";
import { ARCHETYPES, type Archetype } from "@/lib/pareto/archetypes";
import { useCatalog, type GpuOption } from "@/lib/hooks/useCatalog";
import { ModelInput } from "@/components/ui/ModelInput";
import { GpuSystemInput } from "@/components/ui/GpuSystemInput";
import { buildRecipe } from "./recipe";

// ─── Constants ───────────────────────────────────────────────────────────────

const DEFAULT_KV_BW = 25.0;

// ─── Intake pane ─────────────────────────────────────────────────────────────

interface IntakeProps {
  onSubmit: (req: ParetoRequest, sla: number) => void;
  loading: boolean;
}

function IntakePane({ onSubmit, loading }: IntakeProps) {
  const { gpuOptions, modelOptions, isLoading: catalogLoading } = useCatalog();
  const [modelPath, setModelPath] = useState("");
  const [system, setSystem] = useState("");
  const [archetype, setArchetype] = useState<Archetype>(ARCHETYPES[0]);
  const [isl, setIsl] = useState(ARCHETYPES[0].isl);
  const [osl, setOsl] = useState(ARCHETYPES[0].osl);
  const [ttft, setTtft] = useState(2000);
  const [tpot, setTpot] = useState(30);
  const [sla, setSla] = useState(ARCHETYPES[0].sla_tok_per_sec);

  function selectArchetype(a: Archetype) {
    setArchetype(a);
    setIsl(a.isl);
    setOsl(a.osl);
    setSla(a.sla_tok_per_sec);
  }

  function handleSubmit() {
    if (!modelPath || !system) return;
    onSubmit(
      {
        model_path: modelPath,
        system,
        isl,
        osl,
        ttft,
        tpot,
        backend: "vllm",
        kv_transfer_bandwidth_gb_per_second: DEFAULT_KV_BW,
        database_mode: "HYBRID",
      },
      sla
    );
  }

  return (
    <Card style={{ height: "100%" }}>
      <CardTitle>Workload</CardTitle>
      <CardBody>
        <FormGroup label="Model" fieldId="design-model" style={{ marginBottom: 16 }}>
          <ModelInput
            id="design-model"
            model={modelPath}
            onChange={setModelPath}
            modelOptions={modelOptions}
            isLoading={catalogLoading}
          />
        </FormGroup>

        <FormGroup label="GPU system" fieldId="design-system" style={{ marginBottom: 16 }}>
          <GpuSystemInput
            id="design-system"
            value={system}
            onChange={setSystem}
            gpuOptions={gpuOptions}
          />
        </FormGroup>

        <FormGroup label="Workload archetype" fieldId="design-archetype" style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {ARCHETYPES.map((a) => (
              <Label
                key={a.key}
                color={archetype.key === a.key ? "blue" : "grey"}
                onClick={() => selectArchetype(a)}
                style={{ cursor: "pointer" }}
              >
                {a.label}
              </Label>
            ))}
          </div>
          <FormHelperText>
            <HelperText>
              <HelperTextItem>{archetype.description}</HelperTextItem>
            </HelperText>
          </FormHelperText>
        </FormGroup>

        <Split hasGutter style={{ marginBottom: 16 }}>
          <SplitItem>
            <FormGroup label="ISL (tokens)" fieldId="design-isl">
              <NumberInput
                id="design-isl"
                value={isl}
                min={1}
                onMinus={() => setIsl(Math.max(1, isl - 256))}
                onPlus={() => setIsl(isl + 256)}
                onChange={(e) => setIsl(Number((e.target as HTMLInputElement).value))}
              />
            </FormGroup>
          </SplitItem>
          <SplitItem>
            <FormGroup label="OSL (tokens)" fieldId="design-osl">
              <NumberInput
                id="design-osl"
                value={osl}
                min={1}
                onMinus={() => setOsl(Math.max(1, osl - 128))}
                onPlus={() => setOsl(osl + 128)}
                onChange={(e) => setOsl(Number((e.target as HTMLInputElement).value))}
              />
            </FormGroup>
          </SplitItem>
        </Split>

        <Split hasGutter style={{ marginBottom: 16 }}>
          <SplitItem>
            <FormGroup label="TTFT SLA (ms)" fieldId="design-ttft">
              <NumberInput
                id="design-ttft"
                value={ttft}
                min={1}
                onMinus={() => setTtft(Math.max(1, ttft - 100))}
                onPlus={() => setTtft(ttft + 100)}
                onChange={(e) => setTtft(Number((e.target as HTMLInputElement).value))}
              />
            </FormGroup>
          </SplitItem>
          <SplitItem>
            <FormGroup label="TPOT SLA (ms)" fieldId="design-tpot">
              <NumberInput
                id="design-tpot"
                value={tpot}
                min={1}
                onMinus={() => setTpot(Math.max(1, tpot - 5))}
                onPlus={() => setTpot(tpot + 5)}
                onChange={(e) => setTpot(Number((e.target as HTMLInputElement).value))}
              />
            </FormGroup>
          </SplitItem>
        </Split>

        <FormGroup label="Min throughput (tok/s/user)" fieldId="design-sla" style={{ marginBottom: 24 }}>
          <NumberInput
            id="design-sla"
            value={sla}
            min={1}
            onMinus={() => setSla(Math.max(1, sla - 5))}
            onPlus={() => setSla(sla + 5)}
            onChange={(e) => setSla(Number((e.target as HTMLInputElement).value))}
          />
          <FormHelperText>
            <HelperText>
              <HelperTextItem>Frontier filter — only candidates meeting this threshold are highlighted.</HelperTextItem>
            </HelperText>
          </FormHelperText>
        </FormGroup>

        <Button
          variant="primary"
          onClick={handleSubmit}
          isDisabled={!modelPath || !system || loading}
          isLoading={loading}
        >
          {loading ? "Sweeping…" : "Run frontier sweep"}
        </Button>
      </CardBody>
    </Card>
  );
}

// ─── Frontier chart ───────────────────────────────────────────────────────────

interface ChartPaneProps {
  candidates: Candidate[];
  frontierIds: Set<string>;
  sla: number;
  selected: Candidate | null;
  onSelect: (c: Candidate) => void;
  loading: boolean;
}

function FrontierChart({ candidates, frontierIds, sla, selected, onSelect, loading }: ChartPaneProps) {
  const feasible = candidates.filter((c) => c.status === "feasible" && c.objectives);
  const infeasible = candidates.filter((c) => c.status !== "feasible" || !c.objectives);

  const frontierPoints = feasible
    .filter((c) => frontierIds.has(c.candidate_id))
    .sort((a, b) => (a.objectives!.throughput_per_user - b.objectives!.throughput_per_user));

  const toPoint = (c: Candidate) => ({
    x: c.objectives!.throughput_per_user,
    y: c.objectives!.throughput_per_gpu,
    candidate: c,
    name: `${c.candidate_id} (${c.mode}, ${c.used_gpus ?? "?"}× GPU)`,
  });

  if (candidates.length === 0 && !loading) {
    return (
      <EmptyState>
        <EmptyStateBody>Configure a workload and run a sweep to see the Pareto frontier.</EmptyStateBody>
      </EmptyState>
    );
  }

  return (
    <div style={{ position: "relative" }}>
      {loading && (
        <div style={{ position: "absolute", top: 12, right: 16, display: "flex", alignItems: "center", gap: 8, zIndex: 10 }}>
          <Spinner size="sm" />
          <span style={{ fontSize: 13, color: "#54585c" }}>Streaming candidates…</span>
        </div>
      )}
      <Chart
        containerComponent={
          <ChartVoronoiContainer
            labels={({ datum }) => datum.name ?? ""}
            constrainToVisibleArea
          />
        }
        height={340}
        padding={{ top: 24, bottom: 56, left: 72, right: 24 }}
        width={520}
      >
        <ChartAxis
          label="Throughput per user (tok/s/user)"
          style={{ axisLabel: { fontSize: 11, padding: 36 } }}
        />
        <ChartAxis
          dependentAxis
          label="Throughput per GPU (tok/s/GPU)"
          style={{ axisLabel: { fontSize: 11, padding: 52 } }}
        />

        {/* SLA guide line */}
        {sla > 0 && (
          <ChartLine
            data={[
              { x: sla, y: 0 },
              { x: sla, y: Infinity },
            ]}
            style={{ data: { stroke: "#c9190b", strokeDasharray: "4 3", strokeWidth: 1.5 } }}
          />
        )}

        {/* Infeasible / other candidates */}
        {infeasible.length > 0 && (
          <ChartScatter
            data={infeasible
              .filter((c) => c.objectives)
              .map(toPoint)}
            style={{ data: { fill: "#d2d2d2", opacity: 0.6 } }}
            size={4}
          />
        )}

        {/* Feasible non-frontier candidates */}
        {feasible.filter((c) => !frontierIds.has(c.candidate_id)).length > 0 && (
          <ChartScatter
            data={feasible
              .filter((c) => !frontierIds.has(c.candidate_id))
              .map(toPoint)}
            style={{ data: { fill: "#73c5c5", opacity: 0.7 } }}
            size={5}
            events={[
              {
                target: "data",
                eventHandlers: {
                  onClick: (_, props) => {
                    onSelect((props.datum as { candidate: Candidate }).candidate);
                    return [];
                  },
                },
              },
            ]}
          />
        )}

        {/* Pareto frontier points */}
        {frontierPoints.length > 0 && (
          <ChartScatter
            data={frontierPoints.map(toPoint)}
            style={{
              data: {
                fill: ({ datum }) =>
                  selected?.candidate_id === (datum as { candidate: Candidate }).candidate.candidate_id
                    ? "#f0ab00"
                    : "#0066cc",
                stroke: "#fff",
                strokeWidth: 1.5,
              },
            }}
            size={7}
            events={[
              {
                target: "data",
                eventHandlers: {
                  onClick: (_, props) => {
                    onSelect((props.datum as { candidate: Candidate }).candidate);
                    return [];
                  },
                },
              },
            ]}
          />
        )}

        {/* Step line connecting frontier */}
        {frontierPoints.length > 1 && (
          <ChartLine
            data={frontierPoints.map(toPoint)}
            style={{ data: { stroke: "#0066cc", strokeWidth: 1.5, opacity: 0.5 } }}
          />
        )}
      </Chart>

      <div style={{ fontSize: 11, color: "#54585c", marginTop: 4, paddingLeft: 8 }}>
        <span style={{ marginRight: 12 }}>
          <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#0066cc", marginRight: 4 }} />
          Pareto frontier
        </span>
        <span style={{ marginRight: 12 }}>
          <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#73c5c5", marginRight: 4 }} />
          Feasible
        </span>
        <span>
          <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#d2d2d2", marginRight: 4 }} />
          Infeasible
        </span>
        <span style={{ marginLeft: 12, color: "#c9190b" }}>— SLA threshold</span>
      </div>
    </div>
  );
}

// ─── Candidate detail + verdict strip ────────────────────────────────────────

interface DetailProps {
  candidate: Candidate | null;
  recommendation: Candidate | null;
  sla: number;
}

function CandidateDetail({ candidate, recommendation, sla }: DetailProps) {
  const [exportDone, setExportDone] = useState(false);
  const isRecommended = candidate?.candidate_id === recommendation?.candidate_id;
  const clearsSlA =
    candidate?.objectives?.throughput_per_user != null &&
    candidate.objectives.throughput_per_user >= sla;

  function handleExport(format: "vllm" | "llmd") {
    if (!candidate) return;
    const yaml = buildRecipe(candidate, format);
    const blob = new Blob([yaml], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${candidate.candidate_id}-${format}.yaml`;
    a.click();
    URL.revokeObjectURL(url);
    setExportDone(true);
    setTimeout(() => setExportDone(false), 2000);
  }

  if (!candidate) {
    return (
      <EmptyState>
        <EmptyStateBody>Select a candidate on the chart to see details.</EmptyStateBody>
      </EmptyState>
    );
  }

  return (
    <div>
      {/* Verdict strip */}
      <div
        style={{
          padding: "10px 16px",
          borderRadius: 4,
          marginBottom: 16,
          background: isRecommended ? "#f3faf2" : clearsSlA ? "#f0f8ff" : "#fff8f0",
          border: `1px solid ${isRecommended ? "#3e8635" : clearsSlA ? "#0066cc" : "#f0ab00"}`,
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        {isRecommended ? (
          <CheckCircleIcon color="#3e8635" />
        ) : (
          <ExclamationTriangleIcon color={clearsSlA ? "#0066cc" : "#f0ab00"} />
        )}
        <span style={{ fontSize: 13, fontWeight: 500 }}>
          {isRecommended
            ? "Recommended — fewest GPUs clearing the SLA"
            : clearsSlA
            ? "Clears SLA but not the recommended option"
            : "Below SLA threshold"}
        </span>
      </div>

      <DescriptionList isHorizontal style={{ marginBottom: 16 }}>
        <DescriptionListGroup>
          <DescriptionListTerm>Candidate</DescriptionListTerm>
          <DescriptionListDescription>
            <code style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>{candidate.candidate_id}</code>
          </DescriptionListDescription>
        </DescriptionListGroup>
        <DescriptionListGroup>
          <DescriptionListTerm>Mode</DescriptionListTerm>
          <DescriptionListDescription>
            <Label color={candidate.mode === "disagg" || candidate.mode === "disaggregated" ? "blue" : "grey"} isCompact>
              {candidate.mode}
            </Label>
          </DescriptionListDescription>
        </DescriptionListGroup>
        <DescriptionListGroup>
          <DescriptionListTerm>GPUs</DescriptionListTerm>
          <DescriptionListDescription>{candidate.used_gpus ?? "—"}</DescriptionListDescription>
        </DescriptionListGroup>
        <DescriptionListGroup>
          <DescriptionListTerm>Throughput/user</DescriptionListTerm>
          <DescriptionListDescription>
            {candidate.objectives?.throughput_per_user?.toFixed(1) ?? "—"} tok/s
          </DescriptionListDescription>
        </DescriptionListGroup>
        <DescriptionListGroup>
          <DescriptionListTerm>Throughput/GPU</DescriptionListTerm>
          <DescriptionListDescription>
            {candidate.objectives?.throughput_per_gpu?.toFixed(1) ?? "—"} tok/s
          </DescriptionListDescription>
        </DescriptionListGroup>
        {candidate.metrics.ttft_ms != null && (
          <DescriptionListGroup>
            <DescriptionListTerm>TTFT</DescriptionListTerm>
            <DescriptionListDescription>{candidate.metrics.ttft_ms.toFixed(0)} ms</DescriptionListDescription>
          </DescriptionListGroup>
        )}
        {candidate.metrics.tpot_ms != null && (
          <DescriptionListGroup>
            <DescriptionListTerm>TPOT</DescriptionListTerm>
            <DescriptionListDescription>{candidate.metrics.tpot_ms.toFixed(1)} ms</DescriptionListDescription>
          </DescriptionListGroup>
        )}
        <DescriptionListGroup>
          <DescriptionListTerm>Transfer delay modeled</DescriptionListTerm>
          <DescriptionListDescription>{candidate.transfer_delay_modeled ? "Yes" : "No"}</DescriptionListDescription>
        </DescriptionListGroup>
        {candidate.provenance_sources.length > 0 && (
          <DescriptionListGroup>
            <DescriptionListTerm>Provenance</DescriptionListTerm>
            <DescriptionListDescription>{candidate.provenance_sources.join(", ")}</DescriptionListDescription>
          </DescriptionListGroup>
        )}
      </DescriptionList>

      <ActionList>
        <ActionListItem>
          <Button variant="secondary" icon={<DownloadIcon />} onClick={() => handleExport("vllm")} size="sm">
            {exportDone ? "Downloaded" : "Export vLLM recipe"}
          </Button>
        </ActionListItem>
        <ActionListItem>
          <Button variant="secondary" icon={<DownloadIcon />} onClick={() => handleExport("llmd")} size="sm">
            Export llm-d recipe
          </Button>
        </ActionListItem>
      </ActionList>
    </div>
  );
}

// ─── Root ─────────────────────────────────────────────────────────────────────

export function DesignTool() {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [views, setViews] = useState<CandidateViews | null>(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [sla, setSla] = useState(20);
  const cleanupRef = useRef<(() => void) | null>(null);

  const frontier = computeFrontier(candidates);
  const frontierIds = new Set(frontier.map((c) => c.candidate_id));
  const recommendation = pickRecommendation(frontier, sla);

  const handleSubmit = useCallback((req: ParetoRequest, slaTok: number) => {
    // Cancel any in-flight sweep
    cleanupRef.current?.();
    setCandidates([]);
    setViews(null);
    setSelected(null);
    setSla(slaTok);
    setLoading(true);

    const cleanup = streamCandidates(
      req,
      (c) => setCandidates((prev) => [...prev, c]),
      (v) => {
        setViews(v);
        setLoading(false);
      },
      () => setLoading(false)
    );
    cleanupRef.current = cleanup;
  }, []);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "320px 1fr 320px", gap: 16, padding: 16, minHeight: "calc(100vh - 64px)", alignItems: "start" }}>
      {/* Left: Intake */}
      <div>
        <IntakePane onSubmit={handleSubmit} loading={loading} />
      </div>

      {/* Center: Chart */}
      <Card>
        <CardTitle>
          Pareto frontier
          {candidates.length > 0 && (
            <span style={{ fontSize: 12, fontWeight: 400, color: "#54585c", marginLeft: 12 }}>
              {candidates.length} candidates · {frontier.length} on frontier
            </span>
          )}
        </CardTitle>
        <CardBody>
          <FrontierChart
            candidates={candidates}
            frontierIds={frontierIds}
            sla={sla}
            selected={selected}
            onSelect={setSelected}
            loading={loading}
          />
        </CardBody>
      </Card>

      {/* Right: Detail */}
      <Card>
        <CardTitle>
          {selected ? "Selected candidate" : "Candidate detail"}
        </CardTitle>
        <CardBody>
          <CandidateDetail
            candidate={selected ?? recommendation}
            recommendation={recommendation}
            sla={sla}
          />
        </CardBody>
      </Card>
    </div>
  );
}
