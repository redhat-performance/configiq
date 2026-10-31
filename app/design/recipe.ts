import type { Candidate } from "@/lib/types/pareto";

/**
 * Serializes the deploy topology AISimulate computed for a candidate — verbatim.
 *
 * This exports exactly what the SDK produced (deployment mode, GPU count, and the
 * flat parallelism keys from CandidateProvenance.topology). It does NOT synthesize
 * a vLLM launch command or an llm-d manifest — those must come from the SDK's own
 * generator (aisimulate.generator.generate_backend_artifacts, deployment targets
 * vllm / llm-d-kustomize / llm-d-helm), wired server-side. Hand-writing connector
 * names or API versions here would fabricate deployment config the simulator never
 * produced.
 */
export function exportTopology(candidate: Candidate): string {
  const payload = {
    candidate_id: candidate.candidate_id,
    simulator: candidate.simulator,
    mode: candidate.mode,
    used_gpus: candidate.used_gpus,
    topology: candidate.topology,
    objectives: candidate.objectives,
    metrics: candidate.metrics,
    transfer_delay_modeled: candidate.transfer_delay_modeled,
    provenance_sources: candidate.provenance_sources,
  };
  return JSON.stringify(payload, null, 2);
}

/** Human-readable one-line summary of the deploy topology, from real SDK keys. */
export function topologySummary(candidate: Candidate): string {
  const t = candidate.topology;
  if (candidate.mode === "disagg") {
    const p = `prefill TP${t.prefill_tp ?? "?"}×PP${t.prefill_pp ?? "?"}×${t.prefill_replicas ?? "?"}`;
    const d = `decode TP${t.decode_tp ?? "?"}×PP${t.decode_pp ?? "?"}×${t.decode_replicas ?? "?"}`;
    return `${p} · ${d}`;
  }
  return `TP${t.tp ?? "?"} · PP${t.pp ?? "?"}${t.attention_dp ? ` · DP${t.attention_dp}` : ""}`;
}
