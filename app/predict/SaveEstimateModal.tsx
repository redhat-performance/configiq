'use client';

import * as React from 'react';
import { Modal, ModalBody, ModalFooter, ModalHeader, Button, TextInput, FormGroup } from '@patternfly/react-core';

interface SaveEstimateModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (data: { name: string; tags: string; notes: string }) => void;
  defaultName: string;
}

export function SaveEstimateModal({ isOpen, onClose, onSave, defaultName }: SaveEstimateModalProps) {
  const [name, setName] = React.useState(defaultName);
  const [tags, setTags] = React.useState('');
  const [notes, setNotes] = React.useState('');

  React.useEffect(() => {
    if (isOpen) {
      setName(defaultName);
      setTags('');
      setNotes('');
    }
  }, [isOpen, defaultName]);

  const handleSave = () => {
    onSave({ name, tags, notes });
    onClose();
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      variant="small"
    >
      <ModalHeader title="Save estimate" />
      <ModalBody>
        <FormGroup label="Name" isRequired>
          <TextInput
            value={name}
            onChange={(_, val) => setName(val)}
            placeholder="e.g. Llama 3.1 8B · H200 · 97 users"
            style={{ fontFamily: 'var(--font-sans)', fontSize: '14px' }}
          />
        </FormGroup>

        <FormGroup label="Tags (optional)" style={{ marginTop: '16px' }}>
          <TextInput
            value={tags}
            onChange={(_, val) => setTags(val)}
            placeholder="e.g. production, dev, Q3 planning"
            style={{ fontFamily: 'var(--font-sans)', fontSize: '14px' }}
          />
        </FormGroup>

        <FormGroup label="Notes (optional)" style={{ marginTop: '16px' }}>
          <TextInput
            value={notes}
            onChange={(_, val) => setNotes(val)}
            placeholder="e.g. baseline for CIO deck"
            style={{ fontFamily: 'var(--font-sans)', fontSize: '14px' }}
          />
        </FormGroup>
      </ModalBody>
      <ModalFooter>
        <Button variant="primary" onClick={handleSave}>Save estimate</Button>
        <Button variant="link" onClick={onClose}>Cancel</Button>
      </ModalFooter>
    </Modal>
  );
}
