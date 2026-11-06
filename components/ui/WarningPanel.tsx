import { Alert } from '@patternfly/react-core';
import styles from './WarningPanel.module.css';

interface WarningPanelProps {
  warnings: string[];
}

export function WarningPanel({ warnings }: WarningPanelProps) {
  if (warnings.length === 0) return null;
  return (
    <div className={styles.warningsList}>
      {warnings.map((message) => (
        <Alert key={message} isInline variant="warning" title={message} />
      ))}
    </div>
  );
}
