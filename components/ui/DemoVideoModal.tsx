'use client';

import { Modal, ModalBody, ModalFooter, ModalHeader, Button } from '@patternfly/react-core';

interface DemoVideoModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function DemoVideoModal({ isOpen, onClose }: DemoVideoModalProps) {
  return (
    <Modal isOpen={isOpen} onClose={onClose} variant="large">
      <ModalHeader title="ConfigIQ demo" />
      <ModalBody>
        <video
          controls
          preload="metadata"
          src="/demo/configiq-demo.mp4"
          style={{ width: '100%', display: 'block', borderRadius: '4px' }}
        >
          Your browser does not support embedded video.
        </video>
      </ModalBody>
      <ModalFooter>
        <Button variant="link" onClick={onClose}>Close</Button>
      </ModalFooter>
    </Modal>
  );
}
