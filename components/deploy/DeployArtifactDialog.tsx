'use client';

import * as React from 'react';
import {
  Alert,
  Button,
  Checkbox,
  Form,
  FormGroup,
  FormSelect,
  FormSelectOption,
  Modal,
  ModalBody,
  ModalFooter,
  ModalHeader,
  Spinner,
  TextInput,
} from '@patternfly/react-core';
import CheckCircleIcon from '@patternfly/react-icons/dist/esm/icons/check-circle-icon';
import DownloadIcon from '@patternfly/react-icons/dist/esm/icons/download-icon';
import ExclamationCircleIcon from '@patternfly/react-icons/dist/esm/icons/exclamation-circle-icon';
import ExclamationTriangleIcon from '@patternfly/react-icons/dist/esm/icons/exclamation-triangle-icon';

import {
  downloadDeployArtifact,
  fetchDeployTargets,
  renderDeployArtifact,
  saveDeployDownload,
  DeployError,
  type DeployArtifact,
  type DeployRequestInput,
  type DeployTargetInfo,
} from '@/lib/api/deploy';
import { engineRoles, type EngineSection } from '@/lib/api/engine';

import styles from './DeployArtifactDialog.module.css';

export interface DeployArtifactDialogProps {
  isOpen: boolean;
  onClose: () => void;
  /** The limits the result evaluated; null disables rendering. */
  engine: EngineSection | null;
  /** Catalog system identifier the result used. */
  system: string;
  isl: number;
  osl: number;
  concurrency?: number | null;
  /** The model's config.json when the evaluation supplied one. */
  modelConfigData?: Record<string, unknown> | null;
}

const DEFAULT_NAMESPACE = 'configiq';

function errorText(err: unknown): string {
  if (err instanceof DeployError) return err.message;
  return err instanceof Error ? err.message : 'The deployment artifact could not be rendered.';
}

function statusIcon(status: DeployArtifact['checks'][number]['status']): React.ReactNode {
  if (status === 'passed') return <CheckCircleIcon className={styles.passed} aria-hidden="true" />;
  if (status === 'failed') return <ExclamationCircleIcon className={styles.failed} aria-hidden="true" />;
  return <ExclamationTriangleIcon className={styles.skipped} aria-hidden="true" />;
}

function formatField(value: boolean | number | string | null): string {
  if (value == null) return '—';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  return String(value);
}

function EngineSummary({ engine }: { engine: EngineSection }) {
  const roles = engineRoles(engine);
  return (
    <div className={styles.summary}>
      <div className={styles.summaryRow}>
        <span className={styles.summaryLabel}>Engine</span>
        <span className={styles.summaryValue}>
          {engine.backend}
          {engine.backend_version == null ? ' (version not reported)' : ` ${engine.backend_version}`}
        </span>
      </div>
      <div className={styles.summaryRow}>
        <span className={styles.summaryLabel}>Model</span>
        <span className={styles.summaryValue}>{engine.model}</span>
      </div>
      {roles.map(({ role, label, limits }) => (
        <div key={role} className={styles.summaryRow}>
          <span className={styles.summaryLabel}>{label}</span>
          <span className={styles.summaryValue}>
            {formatField(limits.replicas.value)} replica{limits.replicas.value === 1 ? '' : 's'}
            {' · '}TP {formatField(limits.tensor_parallel_size.value)}
            {' · '}PP {formatField(limits.pipeline_parallel_size.value)}
            {' · '}max seqs {formatField(limits.max_num_seqs.value)}
          </span>
        </div>
      ))}
    </div>
  );
}

/**
 * Render a deployment bundle from an evaluated engine section.
 *
 * The dialog gathers only the choices the artifact cannot infer: the target,
 * the install namespace, and any deliberate departure from the target's pin.
 * Everything that sizes the deployment comes from the engine section, and the
 * service re-checks it before answering.
 */
export function DeployArtifactDialog({
  isOpen,
  onClose,
  engine,
  system,
  isl,
  osl,
  concurrency,
  modelConfigData,
}: DeployArtifactDialogProps) {
  const [targets, setTargets] = React.useState<DeployTargetInfo[] | null>(null);
  const [targetsLoading, setTargetsLoading] = React.useState(false);
  const [targetsError, setTargetsError] = React.useState<string | null>(null);
  const [targetId, setTargetId] = React.useState('');
  const [namespace, setNamespace] = React.useState(DEFAULT_NAMESPACE);
  const [namePrefix, setNamePrefix] = React.useState('');
  const [image, setImage] = React.useState('');
  const [allowMismatch, setAllowMismatch] = React.useState(false);
  const [rendering, setRendering] = React.useState(false);
  const [artifact, setArtifact] = React.useState<DeployArtifact | null>(null);
  const [renderError, setRenderError] = React.useState<string | null>(null);
  const [downloading, setDownloading] = React.useState(false);
  const [downloadError, setDownloadError] = React.useState<string | null>(null);
  const inFlight = React.useRef<AbortController | null>(null);
  const downloadRequest = React.useRef<AbortController | null>(null);
  const targetsRequest = React.useRef<AbortController | null>(null);
  // The catalog is fetched once per mount and kept for later opens. Reading it
  // through a ref keeps the fetch effect keyed to `isOpen` alone; depending on
  // the state it sets would abort its own request on the first re-render.
  const targetsRef = React.useRef<DeployTargetInfo[] | null>(null);

  // Cancel unfinished work when the dialog closes or unmounts; a late answer
  // must not overwrite a newer dialog state.
  React.useEffect(
    () => () => {
      inFlight.current?.abort();
      downloadRequest.current?.abort();
    },
    [],
  );

  const resetArtifact = React.useCallback(() => {
    inFlight.current?.abort();
    inFlight.current = null;
    downloadRequest.current?.abort();
    downloadRequest.current = null;
    setRendering(false);
    setDownloading(false);
    setArtifact(null);
    setRenderError(null);
    setDownloadError(null);
  }, []);

  // A different result behind the same dialog must not inherit the previous
  // render: its checks describe limits that are no longer shown.
  const evaluationId = JSON.stringify({
    engine,
    system,
    isl,
    osl,
    concurrency: concurrency ?? null,
    modelConfigData: modelConfigData ?? null,
  });
  React.useEffect(() => {
    resetArtifact();
  }, [evaluationId, resetArtifact]);

  React.useEffect(() => {
    if (!isOpen) {
      targetsRequest.current?.abort();
      targetsRequest.current = null;
      return;
    }
    if (targetsRef.current != null) return;
    const controller = new AbortController();
    targetsRequest.current = controller;
    setTargetsLoading(true);
    setTargetsError(null);
    fetchDeployTargets(controller.signal)
      .then(list => {
        if (controller.signal.aborted) return;
        targetsRef.current = list;
        setTargets(list);
        const firstAvailable = list.find(target => target.available);
        setTargetId(current => current || firstAvailable?.id || list[0]?.id || '');
      })
      .catch(err => {
        if (controller.signal.aborted) return;
        // Leave the catalog unloaded so reopening the dialog retries.
        targetsRef.current = null;
        setTargetsError(errorText(err));
      })
      .finally(() => {
        if (!controller.signal.aborted) setTargetsLoading(false);
      });
    return () => controller.abort();
  }, [isOpen]);

  const selectedTarget = targets?.find(target => target.id === targetId) ?? null;

  const requestInput = (): DeployRequestInput | null => {
    if (engine == null || targetId === '') return null;
    return {
      target: targetId,
      engine,
      system,
      isl,
      osl,
      concurrency: concurrency ?? null,
      namespace,
      namePrefix: namePrefix.trim() || null,
      image: image.trim() || null,
      allowEngineVersionMismatch: allowMismatch,
      modelConfigData: modelConfigData ?? null,
    };
  };

  const handleRender = async () => {
    const input = requestInput();
    if (input == null) return;
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    setRendering(true);
    setRenderError(null);
    setDownloadError(null);
    setArtifact(null);
    try {
      const rendered = await renderDeployArtifact(input, controller.signal);
      if (!controller.signal.aborted) setArtifact(rendered);
    } catch (err: unknown) {
      if (!controller.signal.aborted) setRenderError(errorText(err));
    } finally {
      if (!controller.signal.aborted) setRendering(false);
    }
  };

  const handleDownload = async () => {
    const input = requestInput();
    if (input == null) return;
    downloadRequest.current?.abort();
    const controller = new AbortController();
    downloadRequest.current = controller;
    setDownloading(true);
    setDownloadError(null);
    try {
      const download = await downloadDeployArtifact(input, controller.signal);
      if (!controller.signal.aborted) saveDeployDownload(download);
    } catch (err: unknown) {
      if (!controller.signal.aborted) setDownloadError(errorText(err));
    } finally {
      if (!controller.signal.aborted) setDownloading(false);
    }
  };

  const canRender = engine != null && selectedTarget?.available === true && !rendering;
  const unavailableTargets = (targets ?? []).filter(target => !target.available);

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      variant="medium"
      aria-labelledby="deploy-artifact-title"
    >
      <ModalHeader
        title="Deploy evaluated configuration"
        labelId="deploy-artifact-title"
        description="Render a pinned Kubernetes bundle from the limits this result evaluated. Nothing is re-sized."
      />
      <ModalBody>
        {engine == null ? (
          <Alert
            variant="warning"
            isInline
            title="This result did not report engine limits"
            className={styles.alert}
          >
            A deployment is rendered from the limits the simulation evaluated. Re-run the
            calculation against a simulator that reports them, then save the result again.
          </Alert>
        ) : (
          <>
            <EngineSummary engine={engine} />

            {targetsLoading && <Spinner aria-label="Loading deployment targets" />}
            {targetsError != null && (
              <Alert variant="danger" isInline title="Deployment targets are unavailable" className={styles.alert}>
                {targetsError}
              </Alert>
            )}

            {targets != null && (
              <Form>
                <FormGroup label="Deployment target" fieldId="deploy-target">
                  <FormSelect
                    id="deploy-target"
                    value={targetId}
                    onChange={(_event, value) => {
                      setTargetId(value);
                      resetArtifact();
                    }}
                    aria-label="Deployment target"
                  >
                    {targets.map(target => (
                      <FormSelectOption
                        key={target.id}
                        value={target.id}
                        label={`${target.name} · ${target.simulator}${target.available ? '' : ' — unavailable'}`}
                        isDisabled={!target.available}
                      />
                    ))}
                  </FormSelect>
                  {selectedTarget != null && (
                    <p className={styles.help}>{selectedTarget.description}</p>
                  )}
                </FormGroup>

                {unavailableTargets.length > 0 && (
                  <Alert variant="info" isInline title="Some targets are unavailable here" className={styles.alert}>
                    <ul className={styles.reasons}>
                      {unavailableTargets.map(target => (
                        <li key={target.id}>
                          <strong>{target.name}</strong>: {target.unavailableReason ?? 'This deployment cannot render it.'}
                        </li>
                      ))}
                    </ul>
                  </Alert>
                )}

                <FormGroup label="Namespace" fieldId="deploy-namespace">
                  <TextInput
                    id="deploy-namespace"
                    value={namespace}
                    onChange={(_event, value) => {
                      setNamespace(value);
                      resetArtifact();
                    }}
                  />
                </FormGroup>

                <FormGroup label="Resource name prefix (optional)" fieldId="deploy-name-prefix">
                  <TextInput
                    id="deploy-name-prefix"
                    value={namePrefix}
                    placeholder="configiq"
                    onChange={(_event, value) => {
                      setNamePrefix(value);
                      resetArtifact();
                    }}
                  />
                </FormGroup>

                <FormGroup label="Model-server image override (optional)" fieldId="deploy-image">
                  <TextInput
                    id="deploy-image"
                    value={image}
                    placeholder={selectedTarget?.pins.engine ? `Uses the pinned ${selectedTarget.pins.engine}` : 'Uses the target\u2019s pinned image'}
                    onChange={(_event, value) => {
                      setImage(value);
                      resetArtifact();
                    }}
                  />
                </FormGroup>

                <Checkbox
                  id="deploy-allow-mismatch"
                  label="Render even if the pinned image runs a different engine release"
                  description="The mismatch is recorded in PROVENANCE.md and checked by preflight.sh."
                  isChecked={allowMismatch}
                  onChange={(_event, checked) => {
                    setAllowMismatch(checked);
                    resetArtifact();
                  }}
                />
              </Form>
            )}

            {renderError != null && (
              <Alert variant="danger" isInline title="The artifact was refused" className={styles.alert}>
                {renderError}
              </Alert>
            )}

            {artifact != null && (
              <div className={styles.result}>
                <Alert
                  variant="success"
                  isInline
                  title={`Render gate passed for ${artifact.target}`}
                  className={styles.alert}
                >
                  {artifact.files.length} files · artifact schema {artifact.artifactSchemaVersion}
                </Alert>

                <ul className={styles.checks}>
                  {artifact.checks.map(check => (
                    <li key={check.id} className={styles.check}>
                      {statusIcon(check.status)}
                      <span className={styles.checkId}>{check.id}</span>
                      <span className={styles.checkDetail}>{check.detail}</span>
                    </li>
                  ))}
                </ul>

                {artifact.warnings.length > 0 && (
                  <Alert variant="warning" isInline title="Accepted departures" className={styles.alert}>
                    <ul className={styles.reasons}>
                      {artifact.warnings.map((warning, index) => (
                        <li key={`${index}-${warning}`}>{warning}</li>
                      ))}
                    </ul>
                  </Alert>
                )}

                <div className={styles.pins}>
                  <span className={styles.summaryLabel}>Pinned</span>
                  <span className={styles.summaryValue}>
                    {Object.entries(artifact.pins).map(([key, value]) => `${key}: ${value}`).join(' · ')}
                  </span>
                </div>

                {downloadError != null && (
                  <Alert variant="danger" isInline title="The archive could not be downloaded" className={styles.alert}>
                    {downloadError}
                  </Alert>
                )}
              </div>
            )}
          </>
        )}
      </ModalBody>
      <ModalFooter>
        {artifact != null && (
          <Button
            variant="primary"
            icon={<DownloadIcon />}
            onClick={handleDownload}
            isLoading={downloading}
            isDisabled={downloading}
          >
            Download .zip
          </Button>
        )}
        <Button
          variant={artifact == null ? 'primary' : 'secondary'}
          onClick={handleRender}
          isLoading={rendering}
          isDisabled={!canRender}
        >
          Render artifact
        </Button>
        <Button variant="link" onClick={onClose}>
          Close
        </Button>
      </ModalFooter>
    </Modal>
  );
}
