import { Label } from '@patternfly/react-core'
import type { Simulator } from '@/contexts/SettingsContext'

/** Identifies which simulator produced a result. */
export function SimulatorBadge({ simulator }: { simulator: Simulator }) {
  const isBlis = simulator === 'blis'
  return (
    <Label isCompact color={isBlis ? 'purple' : 'blue'}>
      {isBlis ? 'BLIS' : 'AISimulate'}
    </Label>
  )
}
