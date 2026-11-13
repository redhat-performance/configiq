'use client';

import * as React from 'react';
import { usePathname } from 'next/navigation';
import { getTourForRoute, type HelpTour } from '@/lib/help/tours';
import { ProductTour } from '@/components/ProductTour';
import { recordTourEvent } from '@/lib/help/telemetry';

export interface PageTourContextValue {
  /** Tour registered for the current route, if any. */
  tour: HelpTour | undefined;
  showTour: boolean;
  startTour: () => void;
  /** End the tour from a step, the Skip button, or Escape. */
  completeTour: () => void;
}

export const PageTourContext = React.createContext<PageTourContextValue | null>(null);

/**
 * Owns the manual tour state for the current route and renders the overlay.
 *
 * There is intentionally no auto-open and no persisted "seen" flag: a tour
 * only runs when the user asks for it. The provider sits above the route
 * content so the header button and the overlay share one state instance.
 */
export function PageTourProvider({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const tour = React.useMemo(() => getTourForRoute(pathname), [pathname]);
  const [showTour, setShowTour] = React.useState(false);

  // Leaving the route closes an open tour rather than carrying it across pages.
  React.useEffect(() => {
    setShowTour(false);
  }, [tour?.id]);

  const startTour = React.useCallback(() => {
    if (!tour) return;
    recordTourEvent(tour.id, 'started');
    setShowTour(true);
  }, [tour]);

  const completeTour = React.useCallback(() => {
    if (tour) recordTourEvent(tour.id, 'completed');
    setShowTour(false);
  }, [tour]);

  const value = React.useMemo<PageTourContextValue>(
    () => ({ tour, showTour, startTour, completeTour }),
    [tour, showTour, startTour, completeTour],
  );

  return (
    <PageTourContext.Provider value={value}>
      {children}
      {tour && showTour && <ProductTour steps={tour.steps} onComplete={completeTour} />}
    </PageTourContext.Provider>
  );
}
