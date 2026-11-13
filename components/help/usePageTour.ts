'use client';

import * as React from 'react';
import { PageTourContext, type PageTourContextValue } from './PageTourProvider';

/**
 * No-provider fallback.
 *
 * A page component can legitimately render outside the app shell (unit tests,
 * an embed, a future standalone route). In that case there is no tour to offer,
 * so the button renders nothing instead of throwing and taking the page down.
 */
const NO_TOUR: PageTourContextValue = {
  tour: undefined,
  showTour: false,
  startTour: () => {},
  completeTour: () => {},
};

/** Read the active page tour, or a no-op when no provider is mounted. */
export function usePageTour(): PageTourContextValue {
  return React.useContext(PageTourContext) ?? NO_TOUR;
}
