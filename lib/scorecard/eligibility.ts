import { backendVersionsMatch } from '../validation/benchmark-versions';
import type { MeasuredPoolTopology, ServingMode } from '../validation/benchmark-predictions';

/**
 * Whether one simulator can predict one measured run, and with what inputs.
 *
 * This mirrors the eligibility rules the Benchmark comparison page applies, in
 * the one form a scheduled job can run without a browser. It is driven entirely
 * by the catalogue: no branch here names a simulator, so adding one changes the
 * sweep's results without changing this file.
 *
 * A rejection has two kinds. `ineligible` means the point is outside the
 * comparison: the catalogue lists no such accelerator, model, or backend, so no
 * request could be formed at all. `unsupported` means the parts are listed but
 * this run is beyond what the catalogue says they can model. A request that
 * fails for any other reason is classified by the caller as `inconclusive`,
 * never folded in here.
 */

export interface ScorecardSystem {
  id: string;
  /** Human label the catalogue publishes for the system. */
  label: string;
}

export interface ScorecardBackend {
  id: string;
  versions: string[];
  defaultVersion: string | null;
  /** Releases scoped to a system; an empty list means "not scoped". */
  systems: Record<string, string[]>;
  /** Per-release capabilities the catalogue reports, keyed by release. */
  versionSupport: Record<string, { disaggregated: boolean }>;
  disaggregationRequiresFabric: boolean;
}

export interface ScorecardSimulatorSupport {
  id: string;
  label: string;
  available: boolean;
  version: string | null;
  revision: string | null;
  models: string[];
  modelsExhaustive: boolean;
  systems: ScorecardSystem[];
  backends: ScorecardBackend[];
  networks: string[];
  defaultNetwork: string | null;
}

/** The measured run a point belongs to. */
export interface MeasuredRun {
  model: string;
  accelerator: string;
  isl: number;
  osl: number;
  tp: number;
  backend: string;
  /** Canonical measured release label, or null when the row named none. */
  release: string | null;
  servingMode: ServingMode;
  topology: MeasuredPoolTopology;
}

export interface Eligible {
  simulator: string;
  label: string;
  /** Catalogue system id the measured accelerator resolves to. */
  systemId: string;
  backend: string;
  /** Literal release to send, or null to let the simulator resolve its own. */
  version: string | null;
  /** Whether the predicted release is the one the measured row reports. */
  releaseMatch: boolean;
  requiresFabric: boolean;
  fabric: string | null;
}

export type Eligibility =
  | { status: 'eligible'; plan: Eligible }
  | { status: 'ineligible'; reason: string }
  | { status: 'unsupported'; reason: string };

/** Releases a simulator offers for one backend on one system. */
export function backendVersionsFor(backend: ScorecardBackend, systemId: string | null): string[] {
  if (!systemId) return backend.versions;
  const listed = backend.systems[systemId];
  // Some catalogues scope releases to a system, others report one shared list.
  // An empty system entry means "not scoped", not "unsupported".
  return listed && listed.length > 0 ? listed : backend.versions;
}

/** The catalogue system a measured accelerator label names, if any. */
export function systemForAccelerator(
  systems: readonly ScorecardSystem[],
  accelerator: string,
): ScorecardSystem | null {
  const normalized = accelerator.trim().toLocaleLowerCase();
  if (!normalized) return null;
  return systems.find(system => system.label.toLocaleLowerCase() === normalized)
    ?? systems.find(system => namesMatch(system.label, accelerator))
    ?? null;
}

/** True when `char` does not continue an accelerator name token. */
function isNameBoundary(char: string): boolean {
  return char === '' || !/[a-z0-9]/.test(char);
}

function containsNameToken(haystack: string, needle: string): boolean {
  let index = haystack.indexOf(needle);
  while (index !== -1) {
    const before = index === 0 ? '' : haystack[index - 1];
    const after = index + needle.length >= haystack.length ? '' : haystack[index + needle.length];
    if (isNameBoundary(before) && isNameBoundary(after)) return true;
    index = haystack.indexOf(needle, index + 1);
  }
  return false;
}

/**
 * Whether two accelerator labels name the same part.
 *
 * Token-bounded so a short name such as `B200` cannot match `GB200`, while a
 * display prefix such as `NVIDIA H200` still matches `H200`. Same rule the
 * comparison page applies when it maps a measured accelerator to the catalogue.
 */
export function namesMatch(left: string, right: string): boolean {
  const leftName = left.trim().toLocaleLowerCase();
  const rightName = right.trim().toLocaleLowerCase();
  if (!leftName || !rightName) return false;
  return containsNameToken(leftName, rightName) || containsNameToken(rightName, leftName);
}

/** Decide whether `support` can predict `run`, and with which inputs. */
export function eligibility(
  support: ScorecardSimulatorSupport | undefined,
  run: MeasuredRun,
): Eligibility {
  if (!support) return { status: 'ineligible', reason: 'The catalogue does not list this simulator.' };
  if (!support.available) return { status: 'ineligible', reason: 'Its catalogue capabilities could not be read.' };

  const system = systemForAccelerator(support.systems, run.accelerator);
  if (!system) {
    return {
      status: 'ineligible',
      reason: `${run.accelerator} is not in this simulator's system catalogue.`,
    };
  }
  if (support.modelsExhaustive && !support.models.includes(run.model)) {
    return { status: 'ineligible', reason: 'The catalogue does not list this model for this simulator.' };
  }

  const backend = support.backends.find(option => option.id === run.backend);
  if (!backend) {
    return { status: 'ineligible', reason: `The catalogue does not list ${run.backend} for this simulator.` };
  }

  const listedVersions = backendVersionsFor(backend, system.id);
  const matchedVersion = run.release
    ? listedVersions.find(version => backendVersionsMatch(version, run.release)) ?? null
    : null;
  const effectiveVersion = matchedVersion ?? backend.defaultVersion;
  const versionSupport = effectiveVersion ? backend.versionSupport[effectiveVersion] ?? null : null;

  // A fabric is a disaggregated-serving input. Use the catalogue's own default
  // rather than inventing one, and only when the backend says it needs one.
  const fabric = run.servingMode === 'disagg' ? support.defaultNetwork : null;
  if (run.servingMode === 'disagg') {
    if (versionSupport && !versionSupport.disaggregated) {
      return {
        status: 'unsupported',
        reason: `Release ${effectiveVersion ?? 'default'} does not model prefill/decode disaggregation.`,
      };
    }
    if (backend.disaggregationRequiresFabric && (!fabric || !support.networks.includes(fabric))) {
      return {
        status: 'unsupported',
        reason: 'The catalogue does not name a fabric this simulator can use for disaggregated serving.',
      };
    }
  }

  return {
    status: 'eligible',
    plan: {
      simulator: support.id,
      label: support.label,
      systemId: system.id,
      backend: backend.id,
      // Send the release the row measured only when the catalogue lists it.
      // Otherwise omit the field and let the service resolve its own default:
      // echoing a resolver token back as a version would pin the prediction to
      // something the catalogue never called a release.
      version: matchedVersion,
      releaseMatch: matchedVersion !== null,
      requiresFabric: backend.disaggregationRequiresFabric,
      fabric,
    },
  };
}
