'use client';

import * as React from 'react';
import { Button, Tooltip } from '@patternfly/react-core';
import CubesIcon from '@patternfly/react-icons/dist/esm/icons/cubes-icon';

import type { EngineSection } from '@/lib/api/engine';

import { DeployArtifactDialog } from './DeployArtifactDialog';

export interface DeployArtifactButtonProps {
  /** The limits the displayed result evaluated, or null when it reported none. */
  engine: EngineSection | null;
  /** Catalog system identifier the result used. */
  system: string;
  isl: number;
  osl: number;
  concurrency?: number | null;
  /** The model's config.json when the evaluation supplied one. */
  modelConfigData?: Record<string, unknown> | null;
  /**
   * Why the action is unavailable right now (a calculation in flight, stale
   * inputs, no result), or null when it is available.
   */
  disabledReason: string | null;
  /** Menu-sized variant for a dense result header. */
  size?: 'sm' | 'default';
}

/**
 * Open the deployment dialog for the displayed result.
 *
 * The control only appears when the reader enabled deployment artifacts in
 * Settings; the REST and MCP endpoints are not gated.
 */
export function DeployArtifactButton({
  engine,
  system,
  isl,
  osl,
  concurrency,
  modelConfigData,
  disabledReason,
  size = 'default',
}: DeployArtifactButtonProps) {
  const [isOpen, setIsOpen] = React.useState(false);

  const reason = disabledReason
    ?? (engine == null
      ? 'This result did not report the engine limits a deployment is rendered from.'
      : null);

  const button = (
    <Button
      variant="link"
      size={size}
      icon={<CubesIcon />}
      onClick={() => setIsOpen(true)}
      isDisabled={reason != null}
      aria-label={reason == null ? undefined : `Deploy artifact unavailable. ${reason}`}
    >
      Deploy artifact
    </Button>
  );

  return (
    <>
      {reason == null ? button : (
        <Tooltip content={reason}>
          <span tabIndex={0}>{button}</span>
        </Tooltip>
      )}
      <DeployArtifactDialog
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        engine={engine}
        system={system}
        isl={isl}
        osl={osl}
        concurrency={concurrency ?? null}
        modelConfigData={modelConfigData ?? null}
      />
    </>
  );
}
