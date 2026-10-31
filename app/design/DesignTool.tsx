"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
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
import { fetchFrontier } from "@/lib/api/pareto";
import { ARCHETYPES, type Archetype } from "@/lib/pareto/archetypes";
import { useCatalog, type GpuOption } from "@/lib/hooks/useCatalog";
import { ModelInput } from "@/components/ui/ModelInput";
import { GpuSystemInput } from "@/components/ui/GpuSystemInput";
import { exportTopology, topologySummary, xpydLabel } from "./recipe";

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
  const [maxGpus, setMaxGpus] = useState(8);

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
        max_candidate_gpus: maxGpus,
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

        <Split hasGutter style={{ marginBottom: 24 }}>
          <SplitItem isFilled>
            <FormGroup label="Min throughput (tok/s/user)" fieldId="design-sla">
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
          </SplitItem>
          <SplitItem>
            <FormGroup label="Max GPUs" fieldId="design-maxgpus">
              <NumberInput
                id="design-maxgpus"
                value={maxGpus}
                min={1}
                onMinus={() => setMaxGpus(Math.max(1, maxGpus - 1))}
                onPlus={() => setMaxGpus(maxGpus + 1)}
                onChange={(e) => setMaxGpus(Math.max(1, Number((e.target as HTMLInputElement).value)))}
              />
              <FormHelperText>
                <HelperText>
                  <HelperTextItem>Caps the topology search space.</HelperTextItem>
                </HelperText>
              </FormHelperText>
            </FormGroup>
          </SplitItem>
        </Split>

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

// Validated categorical palette (dataviz skill, light surface):
// frontier/feasible pair passes all six checks (CVD ΔE 17.4, contrast ≥3:1).
const C_FRONTIER = "#0066cc";   // primary — on the frontier
const C_FEASIBLE = "#009596";   // feasible, not on frontier
const C_INFEASIBLE = "#b8bbbe"; // muted — infeasible/unsupported (a state, not a series)
const C_SELECTED = "#f0ab00";   // transient highlight (with white ring)
const C_SLA = "#c9190b";        // SLA threshold reference line
const INK_2 = "#3c3f42";
const INK_CAP = "#54585c";

function useContainerWidth(): [React.RefObject<HTMLDivElement>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(640);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w && w > 0) setWidth(Math.floor(w));
    });
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

function fmtThousands(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : `${Math.round(n)}`;
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6, marginRight: 16 }}>
      <span style={{ width: 10, height: 10, borderRadius: "50%", background: color, display: "inline-block" }} />
      <span style={{ fontSize: 12, color: INK_2 }}>{label}</span>
    </span>
  );
}

function FrontierChart({ candidates, frontierIds, sla, selected, onSelect, loading }: ChartPaneProps) {
  const [wrapRef, width] = useContainerWidth();

  const feasible = candidates.filter((c) => c.status === "feasible" && c.objectives);
  const infeasible = candidates.filter((c) => (c.status !== "feasible") && c.objectives);

  const frontierPoints = feasible
    .filter((c) => frontierIds.has(c.candidate_id))
    .sort((a, b) => a.objectives!.throughput_per_user - b.objectives!.throughput_per_user);
  const feasibleOff = feasible.filter((c) => !frontierIds.has(c.candidate_id));

  const toPoint = (c: Candidate) => ({
    x: c.objectives!.throughput_per_user,
    y: c.objectives!.throughput_per_gpu,
    candidate: c,
    name: `${xpydLabel(c) ?? c.mode} · ${c.used_gpus ?? "?"} GPU · ${Math.round(c.objectives!.throughput_per_user)} tok/s/user · ${fmtThousands(c.objectives!.throughput_per_gpu)} tok/s/GPU`,
  });

  if (candidates.length === 0 && !loading) {
    return (
      <EmptyState>
        <EmptyStateBody>Configure a workload and run a sweep to see the Pareto frontier.</EmptyStateBody>
      </EmptyState>
    );
  }

  // Explicit, data-fit domain with padding. This is the fix for the collapsed
  // y-axis: never feed Infinity to the scale. The SLA line spans the real range.
  const withObj = candidates.filter((c) => c.objectives);
  const xs = withObj.map((c) => c.objectives!.throughput_per_user);
  const ys = withObj.map((c) => c.objectives!.throughput_per_gpu);
  const xMin = Math.min(sla, ...(xs.length ? xs : [0]));
  const xMax = Math.max(sla, ...(xs.length ? xs : [1]));
  const yMin = Math.min(...(ys.length ? ys : [0]));
  const yMax = Math.max(...(ys.length ? ys : [1]));
  const xPad = Math.max((xMax - xMin) * 0.08, 1);
  const yPad = Math.max((yMax - yMin) * 0.1, 1);
  const domain = {
    x: [Math.max(0, xMin - xPad), xMax + xPad] as [number, number],
    y: [Math.max(0, yMin - yPad), yMax + yPad] as [number, number],
  };

  return (
    <div ref={wrapRef} style={{ position: "relative", minHeight: 380 }}>
      {loading && (
        <div style={{ position: "absolute", top: 8, right: 8, display: "flex", alignItems: "center", gap: 8, zIndex: 10 }}>
          <Spinner size="sm" />
          <span style={{ fontSize: 13, color: INK_CAP }}>Running sweep…</span>
        </div>
      )}
      <Chart
        containerComponent={
          <ChartVoronoiContainer labels={({ datum }) => (datum as { name?: string }).name ?? ""} constrainToVisibleArea />
        }
        domain={domain}
        height={360}
        padding={{ top: 20, bottom: 56, left: 76, right: 24 }}
        width={width}
      >
        <ChartAxis
          label="Throughput per user (tok/s/user)"
          style={{
            axisLabel: { fontSize: 12, padding: 38, fill: INK_2 },
            tickLabels: { fontSize: 11, fill: INK_CAP },
            grid: { stroke: "#f0f0f0" },
          }}
        />
        <ChartAxis
          dependentAxis
          label="Throughput per GPU (tok/s/GPU)"
          tickFormat={(t: number) => fmtThousands(t)}
          style={{
            axisLabel: { fontSize: 12, padding: 56, fill: INK_2 },
            tickLabels: { fontSize: 11, fill: INK_CAP },
            grid: { stroke: "#f0f0f0" },
          }}
        />

        {/* SLA threshold — vertical reference line across the real y-domain */}
        {sla > 0 && (
          <ChartLine
            data={[
              { x: sla, y: domain.y[0] },
              { x: sla, y: domain.y[1] },
            ]}
            style={{ data: { stroke: C_SLA, strokeDasharray: "5 4", strokeWidth: 1.5 } }}
          />
        )}

        {/* Frontier connector (drawn under the points) */}
        {frontierPoints.length > 1 && (
          <ChartLine
            data={frontierPoints.map(toPoint)}
            interpolation="stepAfter"
            style={{ data: { stroke: C_FRONTIER, strokeWidth: 2, opacity: 0.35 } }}
          />
        )}

        {/* Infeasible — muted, non-interactive context */}
        {infeasible.length > 0 && (
          <ChartScatter data={infeasible.map(toPoint)} size={4} style={{ data: { fill: C_INFEASIBLE, opacity: 0.7 } }} />
        )}

        {/* Feasible, off-frontier */}
        {feasibleOff.length > 0 && (
          <ChartScatter
            data={feasibleOff.map(toPoint)}
            size={6}
            style={{ data: { fill: C_FEASIBLE, stroke: "#fff", strokeWidth: 1 } }}
            events={[{ target: "data", eventHandlers: { onClick: (_: unknown, p: { datum?: { candidate: Candidate } }) => { if (p.datum) onSelect(p.datum.candidate); return []; } } }]}
          />
        )}

        {/* Pareto frontier — primary */}
        {frontierPoints.length > 0 && (
          <ChartScatter
            data={frontierPoints.map(toPoint)}
            size={(args: { datum?: { candidate: Candidate } }) => (selected?.candidate_id === args.datum?.candidate.candidate_id ? 10 : 8)}
            style={{
              data: {
                fill: (args: { datum?: { candidate: Candidate } }) => (selected?.candidate_id === args.datum?.candidate.candidate_id ? C_SELECTED : C_FRONTIER),
                stroke: "#fff",
                strokeWidth: 2,
              },
            }}
            events={[{ target: "data", eventHandlers: { onClick: (_: unknown, p: { datum?: { candidate: Candidate } }) => { if (p.datum) onSelect(p.datum.candidate); return []; } } }]}
          />
        )}
      </Chart>

      <div style={{ marginTop: 4, paddingLeft: 12, display: "flex", flexWrap: "wrap", alignItems: "center" }}>
        <LegendDot color={C_FRONTIER} label="Pareto frontier" />
        <LegendDot color={C_FEASIBLE} label="Feasible" />
        <LegendDot color={C_INFEASIBLE} label="Infeasible" />
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 16, height: 0, borderTop: `2px dashed ${C_SLA}`, display: "inline-block" }} />
          <span style={{ fontSize: 12, color: INK_2 }}>SLA threshold</span>
        </span>
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

  function handleExport() {
    if (!candidate) return;
    const json = exportTopology(candidate);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${candidate.candidate_id}-topology.json`;
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

      <DescriptionList isHorizontal columnModifier={{ default: "2Col" }} style={{ marginBottom: 16 }}>
        <DescriptionListGroup>
          <DescriptionListTerm>Candidate</DescriptionListTerm>
          <DescriptionListDescription>
            <code style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>{candidate.candidate_id}</code>
          </DescriptionListDescription>
        </DescriptionListGroup>
        <DescriptionListGroup>
          <DescriptionListTerm>Mode</DescriptionListTerm>
          <DescriptionListDescription>
            <Label color={candidate.mode === "disagg" ? "blue" : "grey"} isCompact>
              {candidate.mode === "disagg" ? `disagg · ${xpydLabel(candidate)}` : candidate.mode}
            </Label>
          </DescriptionListDescription>
        </DescriptionListGroup>
        <DescriptionListGroup>
          <DescriptionListTerm>Topology</DescriptionListTerm>
          <DescriptionListDescription>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 12.5 }}>{topologySummary(candidate)}</span>
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
          <Button variant="secondary" icon={<DownloadIcon />} onClick={handleExport} size="sm">
            {exportDone ? "Downloaded" : "Export deploy topology (JSON)"}
          </Button>
        </ActionListItem>
      </ActionList>
      <p style={{ fontSize: 11.5, color: "#54585c", marginTop: 8, fontFamily: "var(--font-mono)" }}>
        Exports the exact topology AISimulate computed. A deployable vLLM / llm-d
        manifest comes from the SDK generator (generate_backend_artifacts), wired server-side.
      </p>
    </div>
  );
}

// ─── Root ─────────────────────────────────────────────────────────────────────

export function DesignTool() {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [views, setViews] = useState<CandidateViews | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [sla, setSla] = useState(20);
  const abortRef = useRef<AbortController | null>(null);

  // Prefer the SDK's authoritative pareto_front; fall back to a client-side
  // non-dominated sort only if the view is missing.
  const frontier = views?.pareto_front?.length
    ? candidates.filter((c) => views.pareto_front.includes(c.candidate_id))
    : computeFrontier(candidates);
  const frontierIds = new Set(frontier.map((c) => c.candidate_id));
  const recommendation = pickRecommendation(frontier, sla);

  const handleSubmit = useCallback(async (req: ParetoRequest, slaTok: number) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setCandidates([]);
    setViews(null);
    setSelected(null);
    setError(null);
    setSla(slaTok);
    setLoading(true);
    try {
      const { candidates: cs, views: vs } = await fetchFrontier(req, controller.signal);
      setCandidates(cs);
      setViews(vs);
    } catch (e) {
      if (!controller.signal.aborted) {
        setError(e instanceof Error ? e.message : "sweep failed");
      }
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "420px minmax(0, 1fr)", gap: 16, padding: 16, minHeight: "calc(100vh - 64px)", alignItems: "start" }}>
      {/* Left: Intake (wider) */}
      <div>
        <IntakePane onSubmit={handleSubmit} loading={loading} />
      </div>

      {/* Right: chart on top, candidate detail below */}
      <div style={{ display: "flex", flexDirection: "column", gap: 16, minWidth: 0 }}>
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
            {error && (
              <div style={{ padding: "10px 16px", marginBottom: 12, borderRadius: 4, background: "#faeae8", border: "1px solid #c9190b", fontSize: 13 }}>
                Sweep failed: {error}
              </div>
            )}
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

        <Card>
          <CardTitle>{selected ? "Selected candidate" : "Candidate detail"}</CardTitle>
          <CardBody>
            <CandidateDetail
              candidate={selected ?? recommendation}
              recommendation={recommendation}
              sla={sla}
            />
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
