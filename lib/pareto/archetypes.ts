export interface Archetype {
  key: string
  label: string
  isl: number
  osl: number
  /** Default tok/s/user SLA for the frontier filter */
  sla_tok_per_sec: number
  description: string
}

export const ARCHETYPES: Archetype[] = [
  {
    key: 'chat',
    label: 'Chat',
    isl: 1024,
    osl: 256,
    sla_tok_per_sec: 20,
    description: 'Interactive chat with short context',
  },
  {
    key: 'coding',
    label: 'Coding',
    isl: 4096,
    osl: 1024,
    sla_tok_per_sec: 15,
    description: 'Code generation and completion',
  },
  {
    key: 'rag',
    label: 'RAG',
    isl: 8192,
    osl: 512,
    sla_tok_per_sec: 15,
    description: 'Retrieval-augmented generation with long context',
  },
  {
    key: 'batch',
    label: 'Batch',
    isl: 2048,
    osl: 2048,
    sla_tok_per_sec: 5,
    description: 'Offline batch inference',
  },
  {
    key: 'agentic',
    label: 'Agentic',
    isl: 16384,
    osl: 512,
    sla_tok_per_sec: 10,
    description: 'Long-context agentic workflows',
  },
]
