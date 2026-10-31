"use client";
import { EmptyState, EmptyStateBody, PageSection, Title } from "@patternfly/react-core";

export default function AccuracyPage() {
  return (
    <PageSection>
      <EmptyState>
        <Title headingLevel="h2" size="lg">Accuracy analysis</Title>
        <EmptyStateBody>Coming soon — per-candidate estimate provenance and measured vs. predicted comparison.</EmptyStateBody>
      </EmptyState>
    </PageSection>
  );
}
