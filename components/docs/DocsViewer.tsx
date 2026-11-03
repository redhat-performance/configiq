"use client";

import React, { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Tabs, Tab, TabContent, TabTitleText, Title, PageSection } from "@patternfly/react-core";
import "./docs-viewer.css";

export type DocSection = {
  key: string;
  title: string;
  markdown: string;
};

/** Internal doc links that switch tabs inside the viewer. */
const INTERNAL_DOCS: Record<string, number> = {
  "user-scenarios.md": 0,
  "mcp-server.md": 1,
  "api.md": 2,
};

const REPO_BLOB = "https://github.com/redhat-performance/configiq/blob/main/";

/** GitHub-style heading slug so in-page anchors keep working. */
function slugify(children: React.ReactNode): string {
  const text = React.Children.toArray(children)
    .map((child) => (typeof child === "string" ? child : ""))
    .join("");
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-");
}

export default function DocsViewer({ docs }: { docs: DocSection[] }) {
  const [activeKey, setActiveKey] = useState<number>(0);

  // Deep link support: /docs?tab=mcp-server opens that tab.
  React.useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const tab = params.get("tab");
    if (tab) {
      const index = docs.findIndex((doc) => doc.key === tab);
      if (index >= 0) setActiveKey(index);
    }
  }, [docs]);

  const components = useMemo(
    () => ({
      h1: ({ children }: any) => <h1 id={slugify(children)}>{children}</h1>,
      h2: ({ children }: any) => <h2 id={slugify(children)}>{children}</h2>,
      h3: ({ children }: any) => <h3 id={slugify(children)}>{children}</h3>,
      h4: ({ children }: any) => <h4 id={slugify(children)}>{children}</h4>,
      a: ({ href, children, ...props }: any) => {
        const raw: string = href ?? "";
        if (raw.startsWith("#")) {
          return (
            <a
              href={raw}
              {...props}
              onClick={(event) => {
                event.preventDefault();
                const target = document.getElementById(raw.slice(1));
                if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
              }}
            >
              {children}
            </a>
          );
        }
        const hashIndex = raw.indexOf("#");
        const file = hashIndex >= 0 ? raw.slice(0, hashIndex) : raw;
        const anchor = hashIndex >= 0 ? raw.slice(hashIndex + 1) : null;
        const tabIndex = INTERNAL_DOCS[file];
        if (tabIndex !== undefined) {
          return (
            <a
              href="#"
              {...props}
              onClick={(event) => {
                event.preventDefault();
                setActiveKey(tabIndex);
                if (anchor) {
                  requestAnimationFrame(() => {
                    const target = document.getElementById(anchor);
                    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
                  });
                }
              }}
            >
              {children}
            </a>
          );
        }
        if (/^https?:\/\//.test(raw)) {
          return (
            <a href={raw} target="_blank" rel="noopener noreferrer" {...props}>
              {children}
            </a>
          );
        }
        // Other repo-relative links (e.g. ../services/..., tested-models.md) go to GitHub.
        const repoPath = file.startsWith("../") ? file.slice(3) : `docs/${file}`;
        return (
          <a
            href={`${REPO_BLOB}${repoPath}${anchor ? `#${anchor}` : ""}`}
            target="_blank"
            rel="noopener noreferrer"
            {...props}
          >
            {children}
          </a>
        );
      },
    }),
    []
  );

  return (
    <PageSection
      padding={{ default: "noPadding" }}
      style={{ backgroundColor: "var(--gc-bg-2, #f5f5f5)", minHeight: "100vh", padding: 0 }}
    >
      <div className="gc-docs-shell">
        <div className="gc-docs-card">
          <Title headingLevel="h1" size="2xl" className="gc-docs-page-title">
            Documentation
          </Title>
          <p className="gc-docs-intro">
            What ConfigIQ is, which tool to use for which ask, how to use the
            MCP server, and the REST API reference.
          </p>
          <Tabs
            activeKey={activeKey}
            onSelect={(_event, key) => setActiveKey(key as number)}
            aria-label="Documentation sections"
          >
            {docs.map((doc, index) => (
              <Tab
                key={doc.key}
                eventKey={index}
                title={<TabTitleText>{doc.title}</TabTitleText>}
                aria-label={doc.title}
              >
                <TabContent
                  eventKey={index}
                  id={`gc-docs-${doc.key}`}
                  aria-label={doc.title}
                  hidden={activeKey !== index}
                >
                  <div className="gc-docs-markdown">
                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
                      {doc.markdown}
                    </ReactMarkdown>
                  </div>
                </TabContent>
              </Tab>
            ))}
          </Tabs>
        </div>
      </div>
    </PageSection>
  );
}
