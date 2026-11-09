import type { Candidate } from "../types/pareto";

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
