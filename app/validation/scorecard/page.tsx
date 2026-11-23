'use client';

import { Suspense } from 'react';
import SimulatorScorecard from './SimulatorScorecard';

export default function Page() {
  return (
    // The scorecard reads its shared-view link with `useSearchParams`.
    <Suspense fallback={null}>
      <SimulatorScorecard />
    </Suspense>
  );
}
