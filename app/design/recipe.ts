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

// Per-role parallelism string, AIS-style: tp2·pp1 (·dp2 ·moe[tp2·ep4] only when >1).
function roleParallelism(
  tp?: number | null, pp?: number | null, dp?: number | null,
  moeTp?: number | null, moeEp?: number | null
): string {
  const parts = [`tp${tp ?? "?"}`, `pp${pp ?? "?"}`];
  if (dp && dp > 1) parts.push(`dp${dp}`);
  if ((moeTp && moeTp > 1) || (moeEp && moeEp > 1)) parts.push(`moe[tp${moeTp ?? 1}·ep${moeEp ?? 1}]`);
  return parts.join("·");
}

/**
 * AISimulate's canonical disagg label: xP:yD = prefill_replicas : decode_replicas.
 * Returns null for aggregated candidates.
 */
export function xpydLabel(candidate: Candidate): string | null {
  if (candidate.mode !== "disagg") return null;
  const t = candidate.topology;
  return `${t.prefill_replicas ?? "?"}P:${t.decode_replicas ?? "?"}D`;
}

/** Full deploy topology, matching AIS's output: xPyD + per-role parallelism. */
export function topologySummary(candidate: Candidate): string {
  const t = candidate.topology;
  if (candidate.mode === "disagg") {
    const p = roleParallelism(t.prefill_tp, t.prefill_pp, t.prefill_attention_dp, t.prefill_moe_tp, t.prefill_moe_ep);
    const d = roleParallelism(t.decode_tp, t.decode_pp, t.decode_attention_dp, t.decode_moe_tp, t.decode_moe_ep);
    return `${xpydLabel(candidate)} — ${t.prefill_replicas ?? "?"}× prefill ${p} · ${t.decode_replicas ?? "?"}× decode ${d}`;
  }
  const agg = roleParallelism(t.tp, t.pp, t.attention_dp, t.moe_tp, t.moe_ep);
  return `${t.replicas ?? "?"}× ${agg}`;
}
