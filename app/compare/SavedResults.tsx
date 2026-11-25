'use client';

import * as React from 'react';

import { CompareResults } from '@/components/comparison/CompareResults';
import { useSettings } from '@/contexts/SettingsContext';
import { useCostings } from '@/lib/hooks/useCostings';

/**
 * The saved-results page reads prices whether or not the reader has costings
 * turned on elsewhere: a saved snapshot with no money figure is a gap this page
 * can fill, and it does not need hosted model prices to do it.
 */
export default function SavedResultsPage() {
  const { pricingSource, preferredCloudProvider } = useSettings();
  const costings = useCostings(true, pricingSource, 'systems');
  const costSources = React.useMemo(() => ({
    gpuCloudRates: costings.gpuCloudRates,
    gpuHardwareCosts: costings.gpuHardwareCosts,
    preferredProvider: preferredCloudProvider,
  }), [costings.gpuCloudRates, costings.gpuHardwareCosts, preferredCloudProvider]);

  return <CompareResults costSources={costSources} />;
}
