'use client';

import * as React from 'react';
import { PageTourContext, type PageTourContextValue } from './PageTourProvider';

/**
 * Read the active page tour. Must be used inside `PageTourProvider`.
 */
export function usePageTour(): PageTourContextValue {
  const context = React.useContext(PageTourContext);
  if (!context) {
    throw new Error('usePageTour must be used inside PageTourProvider');
  }
  return context;
}
