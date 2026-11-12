'use client'

import { Label } from '@patternfly/react-core'
import { useCatalog } from '@/lib/hooks/useCatalog'
import { simulatorDisplayLabel } from '@/lib/simulator-labels'

/**
 * Identifies which simulator produced a result. The label comes from the
 * catalog's simulator list, so a simulator added or renamed by the service
 * needs no change here.
 */
export function SimulatorBadge({ simulator }: { simulator: string }) {
  const { simulators } = useCatalog()
  const label = simulatorDisplayLabel(simulator, simulators.find(entry => entry.id === simulator)?.label ?? simulator)
  return <Label isCompact color="blue">{label}</Label>
}
