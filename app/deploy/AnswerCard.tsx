'use client'

import type { RecommendResult } from '@/lib/api/recommend'
import { latencyVerdict, servedLoadLabel, topologySummary } from './answer-format'
import styles from './deploy.module.css'

/**
 * The answer (docs/07 §4.2): the deployment the simulator recommends for the
 * stated load. Every number comes from the response; the only client-side
 * arithmetic is the comparison against the user's own targets and the
 * free-GPU count.
 */
export function AnswerCard({
  result,
  ttftTarget,
  tpotTarget,
  gpusFree,
  loadUnit,
}: {
  result: RecommendResult
  ttftTarget: number
  tpotTarget: number
  gpusFree: number
  loadUnit: 'users' | 'rps'
}) {
  const needed = result.recommendation.gpusNeeded
  const fits = needed <= gpusFree
  const ttftVerdict = latencyVerdict(result.performance.ttftLatencyMs, ttftTarget)
  const tpotVerdict = latencyVerdict(result.performance.tpotMs, tpotTarget)
  const served = servedLoadLabel(result, loadUnit)

  return (
    <article className={`${styles.answerCard} ${fits ? '' : styles.answerOver}`}>
      <header className={styles.answerHead}>
        <h3 className={styles.answerTitle}>
          {fits ? 'This deployment fits your cluster.' : 'Needs more GPUs than this cluster has free.'}
        </h3>
        {!fits && (
          <p className={styles.answerNote}>
            {needed} GPUs needed, cluster has {gpusFree} free.
          </p>
        )}
      </header>

      <div className={styles.answerHeroRow}>
        <div className={styles.answerHero}>
          <span className={styles.heroValue}>{needed.toLocaleString('en-US')}</span>
          <span className={styles.heroLabel}>of {gpusFree.toLocaleString('en-US')} GPUs free</span>
        </div>
        <div className={styles.targetChips}>
          <TargetChip
            label="Mean TTFT" predicted={result.performance.ttftLatencyMs}
            target={ttftTarget} verdict={ttftVerdict} />
          <TargetChip
            label="Mean TPOT" predicted={result.performance.tpotMs}
            target={tpotTarget} verdict={tpotVerdict} />
        </div>
      </div>

      <p className={styles.servedRow}>
        {served == null
          ? <>The response reported no served load.</>
          : loadUnit === 'users'
            ? <>Serves <strong>{formatServed(served.value)} concurrent users</strong>.</>
            : <>Serves <strong>{formatServed(served.value)} {served.unit}</strong>.</>}
      </p>

      <details className={styles.answerDetails}>
        <summary>Deployment details</summary>
        <dl className={styles.detailList}>
          <div><dt>Deployment</dt><dd>{topologySummary(result)}</dd></div>
          <div><dt>Throughput per GPU</dt>
            <dd>{result.throughput.tokensPerSecondPerGpu.toLocaleString('en-US', { maximumFractionDigits: 1 })} tok/s</dd></div>
          <div><dt>Simulator</dt><dd>{result.simulator}</dd></div>
          <div><dt>Run id</dt><dd className={styles.mono}>{result.requestId}</dd></div>
        </dl>
      </details>

      <p className={styles.disclaimer}>
        Best-effort estimate from the simulator, not a guarantee. Latency is shown against
        your target, not enforced.
      </p>
    </article>
  )
}

function TargetChip({
  label, predicted, target, verdict,
}: {
  label: string
  predicted: number | null
  target: number
  verdict: 'meets' | 'over' | null
}) {
  return (
    <span className={`${styles.targetChip} ${verdict === 'over' ? styles.targetOver : ''}`}>
      <span className={styles.targetLabel}>{label}</span>
      <span className={styles.targetValue}>
        {predicted != null ? predicted.toLocaleString('en-US', { maximumFractionDigits: 1 }) : '—'} ms
      </span>
      {verdict != null && (
        <span className={styles.targetVerdict}>{verdict === 'meets' ? 'meets' : 'over'} {target}</span>
      )}
    </span>
  )
}

function formatServed(value: number): string {
  return value.toLocaleString('en-US', { maximumFractionDigits: 1 })
}
