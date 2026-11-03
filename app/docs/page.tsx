import fs from "fs";
import path from "path";
import DocsViewer, { type DocSection } from "@/components/docs/DocsViewer";

const DOCS = [
  { key: "user-scenarios", title: "User scenarios" },
  { key: "mcp-server", title: "MCP server" },
  { key: "api", title: "REST API" },
];

export default function DocsPage() {
  const sections: DocSection[] = DOCS.map(({ key, title }) => ({
    key,
    title,
    markdown: fs.readFileSync(path.join(process.cwd(), "docs", `${key}.md`), "utf8"),
  }));

  return <DocsViewer docs={sections} />;
}
