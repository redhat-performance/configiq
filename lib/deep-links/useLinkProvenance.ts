'use client';

import * as React from 'react';
import { useSearchParams } from 'next/navigation';
import { useSettings, type InferenceBackend } from '@/contexts/SettingsContext';

/**
 * The provenance fields every shareable view carries.
 *
 * Links are generated with the simulator, backend, and version that produced
 * them, so a recipient can tell a shared view from their own settings instead
 * of quietly reading numbers from a different implementation.
 */
export interface LinkProvenanceValues {
  simulator?: string;
  backend?: InferenceBackend;
  backendVersion?: string;
}

export interface LinkProvenanceState {
  /** Simulator that produced this view, when it differs from this profile's. */
  sharedSimulator: string | null;
  /** Switch this view to the shared simulator; saved settings do not change. */
  acceptSharedSimulator: () => void;
}

/**
 * Apply a shared view's provenance once the page is ready.
 *
 * A different simulator is never switched silently: it is offered through the
 * banner, and only the reader can accept it. When the simulator already
 * matches, the backend and version are applied to this view so the page sends
 * the same request the sender did. Nothing here writes to saved settings.
 *
 * `readValues` is read inside the effect rather than passed by value, because
 * some pages parse their link in an effect of their own and the parsed values
 * do not exist yet on the first render. Call this hook after that effect.
 */
export function useLinkProvenance(
  readValues: () => LinkProvenanceValues | null,
  ready: boolean,
): LinkProvenanceState {
  const { simulator, inferenceBackend, backendVersion, applyViewForSession } = useSettings();
  const search = useSearchParams().toString();
  const [sharedSimulator, setSharedSimulator] = React.useState<string | null>(null);
  const appliedSearch = React.useRef<string | null>(null);
  // The provenance the reader accepted. A later link that omits these fields
  // must not replace the override wholesale and fall back to their saved profile.
  const accepted = React.useRef<LinkProvenanceValues | null>(null);
  const readValuesRef = React.useRef(readValues);
  readValuesRef.current = readValues;

  React.useEffect(() => {
    if (!ready || appliedSearch.current === search) return;
    appliedSearch.current = search;
    const values = readValuesRef.current();
    // A new link is a new view; the previous banner must not linger.
    setSharedSimulator(null);
    if (!values) return;

    if (values.simulator && values.simulator !== simulator) {
      setSharedSimulator(values.simulator);
      return;
    }
    if (values.backend || values.backendVersion) {
      applyViewForSession({
        // Keep a simulator the reader already accepted for this view. The saved
        // default is never pinned here, so changing it still takes effect.
        ...(accepted.current?.simulator ? { simulator: accepted.current.simulator } : {}),
        // Omitted fields keep their current effective values rather than being
        // cleared by the replacement.
        backend: values.backend ?? inferenceBackend,
        backendVersion: values.backendVersion ?? backendVersion,
      });
    }
  }, [ready, search, simulator, inferenceBackend, backendVersion, applyViewForSession]);

  const acceptSharedSimulator = React.useCallback(() => {
    if (!sharedSimulator) return;
    const values = readValuesRef.current();
    const next: LinkProvenanceValues = {
      simulator: sharedSimulator,
      backend: values?.backend ?? inferenceBackend,
      backendVersion: values?.backendVersion ?? backendVersion,
    };
    accepted.current = next;
    applyViewForSession(next);
    setSharedSimulator(null);
  }, [applyViewForSession, sharedSimulator, inferenceBackend, backendVersion]);

  return { sharedSimulator, acceptSharedSimulator };
}
