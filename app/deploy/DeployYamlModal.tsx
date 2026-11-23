'use client'

import * as React from 'react'
import { Button, Modal, ModalBody, ModalFooter, ModalHeader } from '@patternfly/react-core'
import styles from './deploy.module.css'

/**
 * Review panel (docs/08 §7): the objects a deploy would create, rendered for
 * review and copy. Nothing here is applied — the prototype has no cluster
 * connection, and the spec's approval flow is a later phase.
 */
export function DeployYamlModal({
  isOpen, onClose, yaml,
}: {
  isOpen: boolean
  onClose: () => void
  yaml: string
}) {
  const [copied, setCopied] = React.useState(false)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(yaml)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 2000)
    } catch {
      // Clipboard access can be denied; the YAML stays selectable in the modal.
      setCopied(false)
    }
  }

  return (
    <Modal variant="medium" isOpen={isOpen} onClose={onClose}>
      <ModalHeader
        title="Review and deploy"
        description="The objects this deployment would create. Preview only — nothing is applied to a cluster."
      />
      <ModalBody>
        <pre className={styles.yamlPre} tabIndex={0}>{yaml}</pre>
      </ModalBody>
      <ModalFooter>
        <Button key="copy" variant="secondary" onClick={() => void copy()}>
          {copied ? 'Copied' : 'Copy YAML'}
        </Button>
        <Button key="close" variant="link" onClick={onClose}>Close</Button>
      </ModalFooter>
    </Modal>
  )
}
