import ExclamationTriangleIcon from '@patternfly/react-icons/dist/esm/icons/exclamation-triangle-icon';
import styles from './WarningPanel.module.css';

interface WarningPanelProps {
  warnings: string[];
}

export function WarningPanel({ warnings }: WarningPanelProps) {
  if (warnings.length === 0) return null;
  return (
    <div className={styles.warningsList}>
      {warnings.map((message, i) => (
        <div key={i} className={styles.warningItem}>
          <ExclamationTriangleIcon style={{ color: '#f0ab00', flexShrink: 0, marginTop: 1 }} />
          <span>{message}</span>
        </div>
      ))}
    </div>
  );
}
