'use client';

import * as React from 'react';
import Link from 'next/link';
import { useSettings, DEFAULT_PRICING_SOURCE } from '@/contexts/SettingsContext';
import { useCostings, type SourceStatus } from '@/lib/hooks/useCostings';
import { useOnPremProfile, DEFAULT_PROFILE, type OnPremCostProfile } from '@/lib/hooks/useOnPremProfile';
import styles from './Sources.module.css';
import CostAssumptionsEditor from './CostAssumptionsEditor';

// Pricing-feed metadata, served by GET /api/costings/sources. The API owns the
// feed list and labels, so nothing here is hardcoded per feed.
interface ModelSourceOption {
  id: string;
  label: string;
  description: string;
  url: string | null;
}

// Safe default shown only when /sources is unavailable. 'merged' is always a
// valid ?source= value, so the selector stays usable without hardcoding the
// real feed list (which /sources owns when reachable).
const MERGED_FALLBACK: ModelSourceOption = {
  id: 'merged',
  label: 'Merged',
  description: 'All feeds, deduplicated by model id with curated overrides applied',
  url: null,
};

function relativeTime(iso: string | null): string {
  if (!iso) return '—';
  const diff = Date.now() - new Date(iso).getTime();
  const h = Math.floor(diff / 3_600_000);
  const m = Math.floor((diff % 3_600_000) / 60_000);
  if (h > 48) return `${Math.floor(h / 24)}d ago`;
  if (h > 0) return `${h}h ${m}m ago`;
  if (m > 0) return `${m}m ago`;
  return 'just now';
}

function SourceBadge({ status }: { status: SourceStatus }) {
  if (status.last_error && status.stale) {
    return <span className={styles.error}><span className={styles.dot} style={{ background: '#c9190b' }} />Error</span>;
  }
  if (status.stale) {
    return <span className={styles.stale}><span className={styles.dot} style={{ background: '#795600' }} />Stale</span>;
  }
  if (status.last_success) {
    return <span className={styles.fresh}><span className={styles.dot} style={{ background: '#3e8635' }} />Fresh</span>;
  }
  return <span className={styles.unknown}><span className={styles.dot} style={{ background: '#6a6e73' }} />Unknown</span>;
}

function newProfile(): OnPremCostProfile {
  return { ...DEFAULT_PROFILE, id: String(Date.now()), name: 'New profile', hardware: { ...DEFAULT_PROFILE.hardware }, datacenter: { ...DEFAULT_PROFILE.datacenter }, lifecycle: { ...DEFAULT_PROFILE.lifecycle } }
}

export default function Sources() {
  const {
    hydrated, costingsEnabled,
    preferredCloudProvider, setPreferredCloudProvider,
    pricingSource, setPricingSource,
  } = useSettings();
  const { profiles, activeProfileId, setActiveProfile, saveProfile, deleteProfile, exportProfile, importProfile } = useOnPremProfile();
  const [editingProfile, setEditingProfile] = React.useState<OnPremCostProfile | null>(null);
  const [importText, setImportText] = React.useState('');

  // Single source of truth: the shared costings hook (same-origin proxy, cached,
  // and a no-op that fires zero network calls while costings is disabled).
  const costings = useCostings(costingsEnabled, pricingSource);
  const health = costings.health;
  // Treat pre-hydration as loading so the table doesn't flash empty before the
  // persisted toggle/source are known.
  const healthLoading = costings.isLoading || !hydrated;
  const healthError = costings.error;

  // Cloud providers available across all GPU systems (provider.region keys).
  const providers = React.useMemo(() => {
    const keys = new Set<string>();
    for (const rates of costings.gpuCloudRates.values()) {
      for (const k of Object.keys(rates)) keys.add(k);
    }
    return Array.from(keys).sort();
  }, [costings.gpuCloudRates]);

  // Group the flat provider.region keys by provider (the token before the first
  // '.') so a provider with many regions (e.g. Azure) collapses into one
  // accordion row instead of flooding the list and burying single-region
  // providers like vast.ai.
  const providerGroups = React.useMemo(() => {
    const groups = new Map<string, string[]>();
    for (const key of providers) {
      const provider = key.split('.')[0];
      if (!groups.has(provider)) groups.set(provider, []);
      groups.get(provider)!.push(key);
    }
    return Array.from(groups.entries()).sort(([a], [b]) => a.localeCompare(b));
  }, [providers]);

  const [expandedGroups, setExpandedGroups] = React.useState<Record<string, boolean>>({});

  // Keep the group holding the current selection open so the active region is
  // always visible without hunting for it.
  React.useEffect(() => {
    if (preferredCloudProvider) {
      const g = preferredCloudProvider.split('.')[0];
      setExpandedGroups(prev => (prev[g] ? prev : { ...prev, [g]: true }));
    }
  }, [preferredCloudProvider]);

  // Model count per feed, shown against each selectable source. Accumulated
  // across fetches (union with prior values) so a feed's count survives
  // switching to another source, which only loads that one feed's models. The
  // 'merged' total is only trustworthy when the merged feed is loaded (it alone
  // carries every source's models), so it is updated only then.
  const [feedCounts, setFeedCounts] = React.useState<Record<string, number>>({});
  React.useEffect(() => {
    if (costings.models.length === 0) return;
    const perSource: Record<string, number> = {};
    for (const m of costings.models) {
      if (m.source) perSource[m.source] = (perSource[m.source] ?? 0) + 1;
    }
    setFeedCounts(prev => {
      const next = { ...prev, ...perSource };
      if (pricingSource === 'merged') next.merged = costings.models.length;
      return next;
    });
  }, [costings.models, pricingSource]);

  // Available model pricing feeds, served by the API (id + label + description
  // + scrape URL). Fetched once when costings is enabled; the API is the single
  // source of truth so a feed added or retired backend-side is reflected here
  // with no frontend change.
  const [sourceOptions, setSourceOptions] = React.useState<ModelSourceOption[]>([]);
  const [sourcesFromApi, setSourcesFromApi] = React.useState(false);

  React.useEffect(() => {
    if (!costingsEnabled) return;
    let cancelled = false;
    fetch('/api/costings/sources')
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(d => {
        if (cancelled) return;
        const opts = (d.sources ?? []) as ModelSourceOption[];
        if (opts.length > 0) { setSourceOptions(opts); setSourcesFromApi(true); }
      })
      // /sources may be unavailable (an older aicostings deployment lacking the
      // metadata endpoint). Leave the API flag false so effectiveSourceOptions
      // derives the feed list from model provenance instead — the ?source=
      // filter works regardless of whether /sources is deployed.
      .catch(() => { /* fall back to provenance-derived options */ });
    return () => { cancelled = true; };
  }, [costingsEnabled]);

  // Feeds discovered from model provenance, accumulated so a feed stays
  // selectable after switching away from 'merged' (a non-merged fetch only
  // carries that one feed's models). Union-only; never shrinks within a session.
  const [discoveredSources, setDiscoveredSources] = React.useState<string[]>([]);
  React.useEffect(() => {
    const seen = costings.models.map(m => m.source).filter((s): s is NonNullable<typeof s> => s != null);
    if (seen.length === 0) return;
    setDiscoveredSources(prev => {
      const union = new Set([...prev, ...seen]);
      return union.size === prev.length ? prev : Array.from(union).sort();
    });
  }, [costings.models]);

  // What the selector renders: the API list when reachable, otherwise 'merged'
  // plus every feed seen in provenance, so litellm/openrouter remain selectable
  // even when the /sources metadata endpoint is not deployed.
  const effectiveSourceOptions = React.useMemo<ModelSourceOption[]>(() => {
    if (sourcesFromApi && sourceOptions.length > 0) return sourceOptions;
    const opts: ModelSourceOption[] = [MERGED_FALLBACK];
    for (const s of discoveredSources) {
      if (s === 'merged') continue;
      opts.push({
        id: s,
        label: s.charAt(0).toUpperCase() + s.slice(1),
        description: `Hosted model pricing from the ${s} feed`,
        url: null,
      });
    }
    return opts;
  }, [sourcesFromApi, sourceOptions, discoveredSources]);

  // Only reset against the authoritative API list; in degraded mode keep the
  // persisted choice (the ?source= filter accepts it) so a reload before models
  // load doesn't clobber it.
  React.useEffect(() => {
    if (sourcesFromApi && sourceOptions.length > 0 && !sourceOptions.some(s => s.id === pricingSource)) {
      setPricingSource(DEFAULT_PRICING_SOURCE);
    }
  }, [sourcesFromApi, sourceOptions, pricingSource, setPricingSource]);

  // hardware_costs is loaded from bundled seed data, not scraped from a live
  // feed, so it doesn't belong in the live scrape-status table (its coverage is
  // shown by the "GPUs w/ hw cost" summary tile instead).
  const sortedSources = health
    ? Object.entries(health.sources)
        .filter(([name]) => name !== 'hardware_costs')
        .sort(([a], [b]) => a.localeCompare(b))
    : [];

  const staleOrErrored = sortedSources.filter(([, s]) => s.stale || s.last_error).length;
  const allFresh = sortedSources.length > 0 && staleOrErrored === 0;

  // Gate live pricing sections on the Settings toggle, but keep shared cost
  // assumptions editable for tools that remain available without costings.
  if (hydrated && !costingsEnabled) {
    return (
      <div className={styles.page}>
        <div className={styles.header}>
          <h1 className={styles.pageTitle}>Sources</h1>
          <p className={styles.subtitle}>
            Pricing data source status, cloud provider preference, and on-prem cost profiles.
          </p>
        </div>
        <div className={styles.section}>
          <div className={styles.emptyState}>
            Costings features are disabled. Enable them in{' '}
            <Link href="/settings" style={{ color: 'var(--blue)', fontWeight: 600 }}>Settings</Link>{' '}
            to view data sources.
          </div>
        </div>
        <CostAssumptionsEditor />
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <h1 className={styles.pageTitle}>Sources</h1>
        <p className={styles.subtitle}>
          Pricing data source status, cloud provider preference, and on-prem cost profiles.
        </p>
      </div>

      {/* ── Section 1: Pricing data sources ── */}
      <div className={styles.section}>
        <div className={styles.sectionHead}>
          <div>
            <div className={styles.sectionTitle}>Pricing data sources</div>
            <div className={styles.sectionDesc}>
              Live scrape status for each data source powering cost calculations.
            </div>
          </div>
        </div>

        {/* Freshness summary (read-only; the API has no on-demand re-scrape) */}
        {!healthLoading && !healthError && sortedSources.length > 0 && (
          <div className={styles.summaryGrid}>
            <div className={styles.summaryTile}>
              <div className={`${styles.summaryValue} ${allFresh ? styles.summaryValueOk : styles.summaryValueWarn}`}>
                {allFresh ? 'All fresh' : `${staleOrErrored} stale`}
              </div>
              <div className={styles.summaryLabel}>{sortedSources.length} sources</div>
            </div>
            <div className={styles.summaryTile}>
              <div className={styles.summaryValue}>{costings.models.length}</div>
              <div className={styles.summaryLabel}>Models priced</div>
            </div>
            <div className={styles.summaryTile}>
              <div className={styles.summaryValue}>{costings.gpuCloudRates.size}</div>
              <div className={styles.summaryLabel}>GPUs w/ cloud rate</div>
            </div>
            <div className={styles.summaryTile}>
              <div className={styles.summaryValue}>{costings.gpuHardwareCosts.size}</div>
              <div className={styles.summaryLabel}>GPUs w/ hw cost</div>
            </div>
          </div>
        )}

        <table className={styles.table}>
          <thead>
            <tr>
              <th>Source</th>
              <th>Status</th>
              <th>Last success</th>
              <th>Last error</th>
            </tr>
          </thead>
          <tbody>
            {healthLoading && (
              <tr className={styles.loadingRow}>
                <td colSpan={4}>Connecting to aicostings…</td>
              </tr>
            )}
            {healthError && (
              <tr className={styles.loadingRow}>
                <td colSpan={4} style={{ color: '#c9190b' }}>
                  Could not reach the aicostings service — {healthError}
                </td>
              </tr>
            )}
            {!healthLoading && !healthError && sortedSources.map(([name, status]) => (
              <tr key={name}>
                <td><span className={styles.sourceName}>{name}</span></td>
                <td><SourceBadge status={status} /></td>
                <td style={{ color: 'var(--gc-text-2)', fontSize: '12px', fontFamily: 'var(--mono)' }}>
                  {relativeTime(status.last_success)}
                </td>
                <td>
                  {status.last_error && (
                    <div className={styles.errorMsg}>
                      {status.last_error.message}
                      <div style={{ color: '#6a6e73', marginTop: 2 }}>{relativeTime(status.last_error.at)}</div>
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* ── Section 1b: Model pricing source ── */}
      <div className={styles.section}>
        <div className={styles.sectionHead}>
          <div>
            <div className={styles.sectionTitle}>Model pricing source</div>
            <div className={styles.sectionDesc}>
              Which feed to use for hosted LLM API pricing. Merged combines both,
              keyed by model id, with curated overrides applied.
            </div>
          </div>
        </div>
        {effectiveSourceOptions.length === 0 ? (
          <div className={styles.emptyState}>Discovering available feeds…</div>
        ) : (
          <div className={styles.providerGrid}>
            {effectiveSourceOptions.map(opt => (
              <button
                key={opt.id}
                className={`${styles.providerOption} ${pricingSource === opt.id ? styles.providerOptionActive : ''}`}
                onClick={() => setPricingSource(opt.id)}
                title={opt.description}
              >
                <span className={styles.providerDot} />
                {opt.label}
                {feedCounts[opt.id] != null && (
                  <span className={styles.feedCount}>
                    ({feedCounts[opt.id].toLocaleString()} models)
                  </span>
                )}
                {opt.url && (
                  <a
                    href={opt.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={styles.sourceLink}
                    title={`Scrape source: ${opt.url}`}
                    onClick={e => e.stopPropagation()}
                  >
                    ↗
                  </a>
                )}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* ── Section 2: Cloud provider preference ── */}
      <div className={styles.section}>
        <div className={styles.sectionHead}>
          <div>
            <div className={styles.sectionTitle}>Cloud provider preference</div>
            <div className={styles.sectionDesc}>
              Which provider and region to use when calculating self-hosted cloud costs.
              {preferredCloudProvider && <> Currently: <code style={{ fontSize: '12px' }}>{preferredCloudProvider}</code></>}
            </div>
          </div>
          {preferredCloudProvider && (
            <button
              style={{ fontSize: '12px', color: '#6a6e73', background: 'none', border: 'none', cursor: 'pointer', padding: '4px 8px' }}
              onClick={() => setPreferredCloudProvider(null)}
            >
              Clear
            </button>
          )}
        </div>
        {providers.length === 0 ? (
          <div className={styles.emptyState}>
            No cloud rates available — check aicostings connection above.
          </div>
        ) : (
          <div className={styles.providerGroups}>
            {providerGroups.map(([provider, keys]) => {
              const expanded = expandedGroups[provider] ?? false;
              const activeHere = preferredCloudProvider?.split('.')[0] === provider;
              return (
                <div key={provider} className={styles.providerGroup}>
                  <button
                    type="button"
                    className={styles.providerGroupHeader}
                    onClick={() => setExpandedGroups(prev => ({ ...prev, [provider]: !expanded }))}
                    aria-expanded={expanded}
                  >
                    <span className={`${styles.providerGroupChevron} ${expanded ? styles.providerGroupChevronOpen : ''}`}>▸</span>
                    <span className={styles.providerGroupName}>{provider}</span>
                    <span className={styles.providerGroupCount}>
                      {keys.length} {keys.length === 1 ? 'region' : 'regions'}
                    </span>
                    {activeHere && (
                      <span className={styles.providerGroupSelected}>
                        {preferredCloudProvider!.slice(provider.length + 1)}
                      </span>
                    )}
                  </button>
                  {expanded && (
                    <div className={styles.providerGroupBody}>
                      {keys.map(p => (
                        <button
                          key={p}
                          className={`${styles.providerOption} ${preferredCloudProvider === p ? styles.providerOptionActive : ''}`}
                          onClick={() => setPreferredCloudProvider(p)}
                        >
                          <span className={styles.providerDot} />
                          {p.slice(provider.length + 1)}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      <CostAssumptionsEditor />

      {/* ── Section 3: On-prem cost profiles ── */}
      <div className={styles.section}>
        <div className={styles.sectionHead}>
          <div>
            <div className={styles.sectionTitle}>On-prem cost profiles</div>
            <div className={styles.sectionDesc}>
              Customer-specific hardware, power, and staffing costs. Stored locally — exportable as JSON.
            </div>
          </div>
          <button
            style={{ fontSize: '13px', color: 'var(--blue)', background: 'none', border: 'none', cursor: 'pointer', fontWeight: 600 }}
            onClick={() => setEditingProfile(newProfile())}
          >
            + New profile
          </button>
        </div>

        {profiles.length === 0 && !editingProfile && (
          <div className={styles.emptyState}>
            No profiles yet. Create one to model on-prem GPU infrastructure costs.
          </div>
        )}

        {profiles.length > 0 && (
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Profile name</th>
                <th>$/node</th>
                <th>PUE</th>
                <th>Depreciation</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {profiles.map(p => (
                <tr key={p.id}>
                  <td>
                    <span className={styles.sourceName}>{p.name}</span>
                    {activeProfileId === p.id && (
                      <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--blue)', fontWeight: 600 }}>ACTIVE</span>
                    )}
                  </td>
                  <td style={{ fontSize: 13, fontFamily: 'var(--mono)' }}>${p.hardware.serverCostPerNode.toLocaleString()}</td>
                  <td style={{ fontSize: 13, fontFamily: 'var(--mono)' }}>{p.datacenter.pue.toFixed(2)}×</td>
                  <td style={{ fontSize: 13, fontFamily: 'var(--mono)' }}>{p.lifecycle.depreciationYears}yr</td>
                  <td style={{ display: 'flex', gap: 8 }}>
                    <button style={{ fontSize: 12, color: 'var(--blue)', background: 'none', border: 'none', cursor: 'pointer' }}
                      onClick={() => setActiveProfile(activeProfileId === p.id ? null : p.id)}>
                      {activeProfileId === p.id ? 'Deactivate' : 'Use'}
                    </button>
                    <button style={{ fontSize: 12, color: 'var(--blue)', background: 'none', border: 'none', cursor: 'pointer' }}
                      onClick={() => setEditingProfile({ ...p })}>Edit</button>
                    <button style={{ fontSize: 12, color: 'var(--blue)', background: 'none', border: 'none', cursor: 'pointer' }}
                      onClick={() => { const j = exportProfile(p.id); navigator.clipboard.writeText(j) }}>Copy JSON</button>
                    <button style={{ fontSize: 12, color: '#c9190b', background: 'none', border: 'none', cursor: 'pointer' }}
                      onClick={() => deleteProfile(p.id)}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {/* Inline profile editor */}
        {editingProfile && (
          <div style={{ padding: '16px 20px', borderTop: '1px solid var(--gc-line)' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px 24px', marginBottom: 16 }}>
              {(
                [
                  ['Profile name', 'text', editingProfile.name, (v: string) => setEditingProfile(p => p && { ...p, name: v })],
                  ['Server cost per node ($)', 'number', editingProfile.hardware.serverCostPerNode, (v: string) => setEditingProfile(p => p && { ...p, hardware: { ...p.hardware, serverCostPerNode: +v || 0 } })],
                  ['Storage per TB ($)', 'number', editingProfile.hardware.storageCostPerTb, (v: string) => setEditingProfile(p => p && { ...p, hardware: { ...p.hardware, storageCostPerTb: +v || 0 } })],
                  ['Networking total ($)', 'number', editingProfile.hardware.networkingCost, (v: string) => setEditingProfile(p => p && { ...p, hardware: { ...p.hardware, networkingCost: +v || 0 } })],
                  ['Power rate ($/kWh)', 'number', editingProfile.datacenter.powerRatePerKwh, (v: string) => setEditingProfile(p => p && { ...p, datacenter: { ...p.datacenter, powerRatePerKwh: +v || 0 } })],
                  ['PUE', 'number', editingProfile.datacenter.pue, (v: string) => setEditingProfile(p => p && { ...p, datacenter: { ...p.datacenter, pue: +v || 1 } })],
                  ['Rack cost/month ($)', 'number', editingProfile.datacenter.rackCostPerMonth, (v: string) => setEditingProfile(p => p && { ...p, datacenter: { ...p.datacenter, rackCostPerMonth: +v || 0 } })],
                  ['Depreciation (years)', 'number', editingProfile.lifecycle.depreciationYears, (v: string) => setEditingProfile(p => p && { ...p, lifecycle: { ...p.lifecycle, depreciationYears: +v || 5 } })],
                  ['Maintenance (%/yr)', 'number', editingProfile.lifecycle.maintenancePctPerYear, (v: string) => setEditingProfile(p => p && { ...p, lifecycle: { ...p.lifecycle, maintenancePctPerYear: +v || 0 } })],
                  ['Staff FTEs per N nodes', 'number', editingProfile.lifecycle.staffFtesPerNNodes, (v: string) => setEditingProfile(p => p && { ...p, lifecycle: { ...p.lifecycle, staffFtesPerNNodes: +v || 0 } })],
                  ['Staff cost per FTE ($/yr)', 'number', editingProfile.lifecycle.staffCostPerFte, (v: string) => setEditingProfile(p => p && { ...p, lifecycle: { ...p.lifecycle, staffCostPerFte: +v || 0 } })],
                ] as [string, string, string | number, (v: string) => void][]
              ).map(([label, type, value, onChange]) => (
                <label key={label} style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12, fontFamily: 'var(--mono)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--gc-text-2)' }}>
                  {label}
                  <input
                    type={type}
                    style={{ fontFamily: 'var(--mono)', fontSize: 13, padding: '6px 8px', border: '1px solid var(--gc-line)', borderRadius: 4 }}
                    value={String(value)}
                    onChange={e => onChange(e.target.value)}
                  />
                </label>
              ))}
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button style={{ fontSize: 13, padding: '6px 14px', background: 'var(--blue)', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer', fontWeight: 600 }}
                onClick={() => { saveProfile(editingProfile!); setEditingProfile(null) }}>Save</button>
              <button style={{ fontSize: 13, padding: '6px 14px', background: 'none', border: '1px solid var(--gc-line)', borderRadius: 4, cursor: 'pointer' }}
                onClick={() => setEditingProfile(null)}>Cancel</button>
            </div>
          </div>
        )}

        {/* Import */}
        <div style={{ padding: '12px 20px', borderTop: '1px solid var(--gc-line)', display: 'flex', gap: 8, alignItems: 'center' }}>
          <input
            style={{ flex: 1, fontFamily: 'var(--mono)', fontSize: 12, padding: '6px 8px', border: '1px solid var(--gc-line)', borderRadius: 4 }}
            placeholder="Paste JSON to import a profile…"
            value={importText}
            onChange={e => setImportText(e.target.value)}
          />
          <button
            style={{ fontSize: 13, padding: '6px 14px', background: 'none', border: '1px solid var(--gc-line)', borderRadius: 4, cursor: 'pointer', whiteSpace: 'nowrap' }}
            onClick={() => { importProfile(importText); setImportText('') }}
          >
            Import
          </button>
        </div>
      </div>
    </div>
  );
}
