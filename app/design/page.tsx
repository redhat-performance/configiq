"use client";
import { PageSection } from "@patternfly/react-core";
import { DesignTool } from "./DesignTool";

export default function DesignPage() {
  return (
    <PageSection style={{ padding: 0, background: "#f5f5f5", minHeight: "100vh" }}>
      <DesignTool />
    </PageSection>
  );
}
