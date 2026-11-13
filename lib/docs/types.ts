export interface DocTocEntry {
  id: string;
  title: string;
  level: number;
}

export interface GeneratedDoc {
  slug: string;
  title: string;
  html: string;
  toc: DocTocEntry[];
}
