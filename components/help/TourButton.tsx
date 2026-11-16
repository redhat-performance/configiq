'use client';

import * as React from 'react';
import { Button } from '@patternfly/react-core';
import CompassIcon from '@patternfly/react-icons/dist/esm/icons/compass-icon';
import { usePageTour } from './usePageTour';

/**
 * Header button that starts the tour registered for the current route.
 * Renders nothing when the route has no tour, so pages do not need a flag.
 */
export function TourButton() {
  const { tour, startTour } = usePageTour();
  if (!tour) return null;

  return (
    <Button variant="secondary" icon={<CompassIcon />} onClick={startTour}>
      Take a tour
    </Button>
  );
}
