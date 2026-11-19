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
  /**
   * `proposal` marks a link built from a page's own link contract with the
   * values the question named. It is inputs only — opening it still requires
   * the user to press Calculate — and the panel says which fields it sets.
   * `page` marks a link into a tool page: an action to take, not something to
   * read. Documentation links carry neither, so only an action is styled as
   * the primary one.
   */
  kind?: 'proposal' | 'page';
}

export type AssistantErrorCode =
  | 'assistant_disabled'
  | 'assistant_busy'
  | 'assistant_unavailable'
  | 'invalid_request';

/**
 * `suppressed` means at least one number in the draft prose could not be traced
 * to the retrieved sections or a tool result. `incomplete` means generation
 * stopped before the model finished its turn. In both cases the prose is not
 * shown; the evidence and links are.
 */
export type AssistantResultStatus = 'answered' | 'suppressed' | 'incomplete';

export type AssistantEvent =
  | { type: 'retrieval'; sections: RetrievedSection[] }
  | { type: 'tool'; name: string; ok: boolean; summary: string }
  | { type: 'token'; text: string }
  | {
      type: 'result';
      status: AssistantResultStatus;
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
  /** The model the page is currently set to, so a proposal can carry it. */
  model?: string;
  /** The GPU system the page is currently set to. */
  system?: string;
}
