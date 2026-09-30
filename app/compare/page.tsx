'use client';

import * as React from 'react';
import {
  PageSection,
  Title,
  Content,
  Button,
  Card,
  CardBody,
  Alert,
} from '@patternfly/react-core';
import Link from 'next/link';
import { getSavedEstimates, deleteEstimate, clearAllEstimates, type SavedEstimate } from '@/lib/saved-estimates';
import { ComparisonRow } from './ComparisonRow';
import styles from './compare.module.css';

// Mock data for development
const MOCK_DATA: SavedEstimate[] = [
  {
    id: '1',
    name: 'Qwen 2.5 32B',
    tags: 'production',
    notes: '',
    savedAt: new Date().toISOString(),
    model: 'Qwen/Qwen2.5-32B-Instruct',
    gpu: 'NVIDIA H100 80GB',
    inputs: {
      isl: 8000,
      osl: 0,
      concurrentUsers: 30,
      weightPrecision: 'FP16',
      kvCachePrecision: 'FP16',
    },
    results: {
      gpusRequired: 8,
      tpSize: 1,
      ppSize: 1,
      replicas: 8,
      weightMemoryGB: 64,
      kvCachePerUserGB: 2.0,
      kvCacheTotalGB: 9.1,
      kvCacheMBPerToken: 248,
      kvCategory: 'KV-1',
      kvCategoryLabel: 'Standard Dense',
      cloudCostMonthly: 175300,
      cloudCost5Year: 10518000,
      selfHostedCostMonthly: 3750,
      selfHostedCost5Year: 225000,
    },
  },
  {
    id: '2',
    name: 'Llama 3.1 70B',
    tags: 'staging',
    notes: 'baseline for CIO deck',
    savedAt: new Date().toISOString(),
    model: 'meta-llama/Llama-3.1-70B-Instruct',
    gpu: 'NVIDIA H100 80GB',
    inputs: {
      isl: 8000,
      osl: 0,
      concurrentUsers: 30,
      weightPrecision: 'FP16',
      kvCachePrecision: 'FP16',
    },
    results: {
      gpusRequired: 4,
      tpSize: 2,
      ppSize: 1,
      replicas: 2,
      weightMemoryGB: 140,
      kvCachePerUserGB: 2.5,
      kvCacheTotalGB: 146.7,
      kvCacheMBPerToken: 310,
      kvCategory: 'KV-1',
      kvCategoryLabel: 'Standard Dense',
      cloudCostMonthly: 87650,
      cloudCost5Year: 5259000,
      selfHostedCostMonthly: 1867,
      selfHostedCost5Year: 112000,
    },
  },
  {
    id: '3',
    name: 'Gemma 4 31B',
    tags: '',
    notes: '',
    savedAt: new Date().toISOString(),
    model: 'google/gemma-4-31b-it',
    gpu: 'NVIDIA H100 80GB',
    inputs: {
      isl: 2000,
      osl: 0,
      concurrentUsers: 4,
      weightPrecision: 'FP16',
      kvCachePrecision: 'FP16',
    },
    results: {
      gpusRequired: 8,
      tpSize: 1,
      ppSize: 1,
      replicas: 8,
      weightMemoryGB: 62,
      kvCachePerUserGB: 0.054,
      kvCacheTotalGB: 143.3,
      kvCacheMBPerToken: 27,
      kvCategory: 'KV-1',
      kvCategoryLabel: 'Standard Dense',
      cloudCostMonthly: 368400,
      cloudCost5Year: 22104000,
      selfHostedCostMonthly: 3333,
      selfHostedCost5Year: 200000,
    },
  },
  {
    id: '4',
    name: 'DeepSeek-R1-32B',
    tags: 'dev',
    notes: 'lowest cost option',
    savedAt: new Date().toISOString(),
    model: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    gpu: 'NVIDIA H100 80GB',
    inputs: {
      isl: 2000,
      osl: 0,
      concurrentUsers: 4,
      weightPrecision: 'FP16',
      kvCachePrecision: 'FP16',
    },
    results: {
      gpusRequired: 1,
      tpSize: 1,
      ppSize: 1,
      replicas: 1,
      weightMemoryGB: 64,
      kvCachePerUserGB: 0.497,
      kvCacheTotalGB: 7.2,
      kvCacheMBPerToken: 248,
      kvCategory: 'KV-2',
      kvCategoryLabel: 'MLA',
      cloudCostMonthly: 46000,
      cloudCost5Year: 2760000,
      selfHostedCostMonthly: 417,
      selfHostedCost5Year: 25000,
    },
  },
];

export default function ComparePage() {
  const [estimates, setEstimates] = React.useState<SavedEstimate[]>([]);
  const [showCioSummary, setShowCioSummary] = React.useState(false);
  const [toast, setToast] = React.useState('');

  React.useEffect(() => {
    // Use mock data for now, will switch to real data later
    const saved = getSavedEstimates();
    setEstimates(saved.length > 0 ? saved : MOCK_DATA);
  }, []);

  const handleDelete = (id: string) => {
    if (confirm('Remove this estimate from comparison?')) {
      deleteEstimate(id);
      setEstimates(estimates.filter(e => e.id !== id));
    }
  };

  const handleClearAll = () => {
    if (confirm('Clear all saved estimates? This cannot be undone.')) {
      clearAllEstimates();
      setEstimates([]);
    }
  };

  const showToast = (message: string) => {
    setToast(message);
    setTimeout(() => setToast(''), 3000);
  };

  const copyForSheets = () => {
    // Generate TSV
    const headers = ['Metric', ...estimates.map(e => e.name)];
    const rows = [
      ['Model', ...estimates.map(e => e.model)],
      ['GPU', ...estimates.map(e => e.gpu)],
      ['Users', ...estimates.map(e => e.inputs.concurrentUsers.toString())],
      ['Context', ...estimates.map(e => `${e.inputs.isl + e.inputs.osl}`)],
      ['Precision', ...estimates.map(e => e.inputs.weightPrecision)],
      ['', ...estimates.map(() => '')], // Separator
      ['KV / user (GB)', ...estimates.map(e => e.results.kvCachePerUserGB.toFixed(3))],
      ['KV pool (GB)', ...estimates.map(e => e.results.kvCacheTotalGB.toFixed(1))],
      ['MB / token', ...estimates.map(e => e.results.kvCacheMBPerToken.toFixed(0))],
      ['', ...estimates.map(() => '')], // Separator
      ['GPUs needed', ...estimates.map(e => e.results.gpusRequired.toString())],
      ['TP size', ...estimates.map(e => e.results.tpSize.toString())],
      ['PP size', ...estimates.map(e => e.results.ppSize.toString())],
      ['Replicas', ...estimates.map(e => e.results.replicas.toString())],
      ['', ...estimates.map(() => '')], // Separator
      ['Self-hosted (5yr)', ...estimates.map(e => `$${(e.results.selfHostedCost5Year / 1000).toFixed(0)}K`)],
      ['Cloud (5yr)', ...estimates.map(e => `$${(e.results.cloudCost5Year / 1000).toFixed(0)}K`)],
      ['Cloud/Self multiplier', ...estimates.map(e => `${(e.results.cloudCost5Year / e.results.selfHostedCost5Year).toFixed(1)}×`)],
    ];

    const tsv = [headers, ...rows].map(row => row.join('\t')).join('\n');
    navigator.clipboard.writeText(tsv);
    showToast('Copied — paste into Google Sheets');
  };

  const copyCioSummary = () => {
    const summary = generateCioSummary();
    navigator.clipboard.writeText(summary);
    showToast('Summary copied');
  };

  const generateCioSummary = () => {
    if (estimates.length === 0) return '';

    const lowest = estimates.reduce((min, e) =>
      e.results.gpusRequired < min.results.gpusRequired ? e : min
    );
    const highest = estimates.reduce((max, e) =>
      e.results.kvCachePerUserGB > max.results.kvCachePerUserGB ? e : max
    );

    const multipliers = estimates.map(e => e.results.cloudCost5Year / e.results.selfHostedCost5Year);
    const minMult = Math.min(...multipliers);
    const maxMult = Math.max(...multipliers);

    return `Comparing ${estimates.length} configurations, ${lowest.name} requires the fewest GPUs (${lowest.results.gpusRequired}) and lowest 5-year self-hosted cost ($${(lowest.results.selfHostedCost5Year / 1000).toFixed(0)}K). ${highest.name} has the highest KV cache demand (${highest.results.kvCachePerUserGB.toFixed(1)} GB/user). Cloud costs run ${minMult.toFixed(0)}–${maxMult.toFixed(0)}× higher than self-hosted across all configurations.`;
  };

  if (estimates.length === 0) {
    return (
      <PageSection>
        <div style={{ maxWidth: '600px', margin: '80px auto', textAlign: 'center' }}>
          <div style={{ fontSize: '48px', marginBottom: '24px' }}>📊</div>
          <Title headingLevel="h1" size="2xl" style={{ marginBottom: '12px' }}>
            No saved estimates yet
          </Title>
          <Content component="p" style={{ marginBottom: '32px', color: '#6a6e73', fontSize: '15px' }}>
            Run Predict performance and click &ldquo;Save estimate&rdquo; to compare configurations here.
          </Content>
          <Link href="/predict">
            <Button variant="primary" size="lg">
              → Go to Predict performance
            </Button>
          </Link>
        </div>
      </PageSection>
    );
  }

  return (
    <>
      <div style={{ padding: '20px 24px 0' }}>
        <h1 className={styles.pageTitle}>Compare saved results</h1>
        <p className={styles.subtitle}>Compare workloads side by side. Save from Predict performance.</p>
      </div>

      <PageSection>
        {/* Toolbar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px', alignItems: 'center' }}>
          <div style={{ display: 'flex', gap: '8px' }}>
            <Button variant="secondary" size="sm">Export PDF</Button>
            <Button variant="secondary" size="sm">Download Excel</Button>
            <Button variant="secondary" size="sm" onClick={copyForSheets}>Copy for Sheets</Button>
            <Button variant="secondary" size="sm">Export to Sheets</Button>
            <Button variant="danger" size="sm" onClick={handleClearAll}>Clear all</Button>
          </div>
          <Link href="/predict">
            <Button variant="primary">+ Add estimate</Button>
          </Link>
        </div>

        {/* CIO Summary */}
        <Card isCompact style={{ marginBottom: '20px' }}>
          <CardBody>
            <div
              onClick={() => setShowCioSummary(!showCioSummary)}
              style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
            >
              <span style={{ fontWeight: 600, fontSize: '14px' }}>Executive summary</span>
              <span>{showCioSummary ? '▾' : '▸'}</span>
            </div>
            {showCioSummary && (
              <div style={{ marginTop: '16px' }}>
                <Content component="p" style={{ marginBottom: '16px', lineHeight: 1.6 }}>
                  {generateCioSummary()}
                </Content>
                <Button variant="secondary" size="sm" onClick={copyCioSummary}>
                  Copy for CIO deck
                </Button>
              </div>
            )}
          </CardBody>
        </Card>

        {/* Comparison Table */}
        <div className={styles.tableWrapper}>
          <table className={styles.comparisonTable}>
            <thead>
              <tr>
                <th className={styles.metricCol}>METRIC</th>
                {estimates.map(est => (
                  <th key={est.id} className={styles.estimateCol}>
                    <div className={styles.colHeader}>
                      <div className={styles.colName}>{est.name}</div>
                      <div className={styles.colMeta}>
                        {est.inputs.weightPrecision} · {est.inputs.concurrentUsers} users · {est.inputs.isl + est.inputs.osl} ctx
                      </div>
                      <div className={styles.colActions}>
                        <button className={styles.actionBtn}>Load</button>
                        <button className={styles.actionBtn}>Verify</button>
                        <button className={styles.actionBtn}>Share</button>
                        <button className={styles.actionBtn} onClick={() => handleDelete(est.id)}>×</button>
                      </div>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
<tbody>
              {/* INPUTS SECTION */}
              <tr className={styles.sectionHeader}>
                <td colSpan={estimates.length + 1}>
                  <div className={styles.sectionLabel} style={{ borderLeft: '4px solid #0066cc' }}>
                    INPUTS
                  </div>
                </td>
              </tr>
              <ComparisonRow
                label="Model"
                estimates={estimates}
                getValue={(e) => e.model.split('/').pop() || e.model}
              />
              <ComparisonRow
                label="GPU"
                estimates={estimates}
                getValue={(e) => e.gpu.replace('NVIDIA ', '').replace('AMD ', '')}
              />
              <ComparisonRow
                label="Users"
                estimates={estimates}
                getValue={(e) => e.inputs.concurrentUsers}
                showBar
                barColor="#0066cc"
              />
              <ComparisonRow
                label="Context"
                estimates={estimates}
                getValue={(e) => e.inputs.isl + e.inputs.osl}
                format={(v) => `${v} tok`}
                showBar
                barColor="#0066cc"
              />
              <ComparisonRow
                label="Precision"
                estimates={estimates}
                getValue={(e) => e.inputs.weightPrecision}
              />

              {/* KV CACHE SECTION */}
              <tr className={styles.sectionHeader}>
                <td colSpan={estimates.length + 1}>
                  <div className={styles.sectionLabel} style={{ borderLeft: '4px solid #f4a523' }}>
                    KV CACHE
                  </div>
                </td>
              </tr>
              <ComparisonRow
                label="KV / user"
                estimates={estimates}
                getValue={(e) => e.results.kvCachePerUserGB}
                format={(v) => v >= 1 ? `${v.toFixed(1)} GB` : `${(v * 1000).toFixed(0)} MB`}
                showBar
                barColor="#f4a523"
              />
              <ComparisonRow
                label="KV pool (GB)"
                estimates={estimates}
                getValue={(e) => e.results.kvCacheTotalGB}
                format={(v) => `${v.toFixed(1)} GB`}
                showBar
                barColor="#f4a523"
              />
              <ComparisonRow
                label="MB / token"
                estimates={estimates}
                getValue={(e) => e.results.kvCacheMBPerToken}
                format={(v) => `${v.toFixed(0)} KB`}
                showBar
                barColor="#f4a523"
              />

              {/* GPU SIZING SECTION */}
              <tr className={styles.sectionHeader}>
                <td colSpan={estimates.length + 1}>
                  <div className={styles.sectionLabel} style={{ borderLeft: '4px solid #7c3aed' }}>
                    GPU SIZING
                  </div>
                </td>
              </tr>
              <ComparisonRow
                label="GPUs needed"
                estimates={estimates}
                getValue={(e) => e.results.gpusRequired}
                showBar
                barColor="#7c3aed"
                lowerIsBetter
              />
              <ComparisonRow
                label="TP size"
                estimates={estimates}
                getValue={(e) => e.results.tpSize}
              />
              <ComparisonRow
                label="PP size"
                estimates={estimates}
                getValue={(e) => e.results.ppSize}
              />
              <ComparisonRow
                label="Replicas"
                estimates={estimates}
                getValue={(e) => e.results.replicas}
                showBar
                barColor="#7c3aed"
              />

              {/* COST SECTION */}
              <tr className={styles.sectionHeader}>
                <td colSpan={estimates.length + 1}>
                  <div className={styles.sectionLabel} style={{ borderLeft: '4px solid #3d7317' }}>
                    COST · 5-YEAR TOTAL
                  </div>
                </td>
              </tr>
              <ComparisonRow
                label="Self-hosted total"
                estimates={estimates}
                getValue={(e) => e.results.selfHostedCost5Year}
                format={(v) => `$${(v / 1000).toFixed(0)}K`}
                showBar
                barColor="#3d7317"
                lowerIsBetter
              />
              <ComparisonRow
                label="Cloud total"
                estimates={estimates}
                getValue={(e) => e.results.cloudCost5Year}
                format={(v) => `$${(v / 1000).toFixed(0)}K`}
                showBar
                barColor="#3d7317"
                lowerIsBetter
              />
              <ComparisonRow
                label="Cloud/self-hosted multiplier"
                estimates={estimates}
                getValue={(e) => e.results.cloudCost5Year / e.results.selfHostedCost5Year}
                format={(v) => `${v.toFixed(1)}×`}
                showBar
                barColor="#f4a523"
              />
            </tbody>
          </table>
        </div>

        {toast && (
          <div className={styles.toast}>
            {toast}
          </div>
        )}
      </PageSection>
    </>
  );
}
