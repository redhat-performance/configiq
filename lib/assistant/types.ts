/**
 * Shared assistant types.
 *
 * The route streams newline-delimited JSON events rather than a single JSON
 * body so long answers can render while they are produced. Every stream ends
 * with exactly one `result` or `error` event; a `result` is only ever emitted
 * after the deterministic verifier has run.
 */

export interface RetrievedSection {
  id: string;
  title: string;
  section: string;
  url: string;
  body: string;
  score: number;
}

export interface AssistantCitation {
  /** `/docs/<slug>#<anchor>` */
  url: string;
  title: string;
  section: string;
}

export interface AssistantLink {
  label: string;
  href: string;
  description?: string;
}

export type AssistantErrorCode =
  | 'assistant_disabled'
  | 'assistant_busy'
  | 'assistant_unavailable'
  | 'invalid_request';

export type AssistantEvent =
  | { type: 'retrieval'; sections: RetrievedSection[] }
  | { type: 'tool'; name: string; ok: boolean; summary: string }
  | { type: 'token'; text: string }
  | {
      type: 'result';
      /**
       * `suppressed` means at least one number in the draft prose could not be
       * traced to the retrieved sections or a tool result. The prose is not
       * shown; the evidence and links are.
       */
      status: 'answered' | 'suppressed';
      answer: string;
      citations: AssistantCitation[];
      links: AssistantLink[];
      evidence: string[];
      unsupportedNumbers: string[];
      model: string;
    }
  | { type: 'error'; code: AssistantErrorCode; message: string };

export interface AssistantRequest {
  question: string;
  /** Route the user asked from, used for suggestions and link defaults. */
  route?: string;
  /** Simulator the user has selected, forwarded as provenance on links. */
  simulator?: string;
  backend?: string;
  backendVersion?: string;
}
