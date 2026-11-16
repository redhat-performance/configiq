/**
 * Auditing a page's request against its link contract.
 *
 * A shared view is only complete if every value the page submits to the API is
 * either carried by the link or deliberately left out. Totality in the accessor
 * table proves the first half — each contract field reaches page state — but
 * says nothing about a request field that no contract field produces, which is
 * how a new input silently misses the shared view: the form works, the request
 * is right, and the link quietly describes a different view.
 *
 * This is the second half. A page declares, once, how each request key relates
 * to its contract, and the audit fails when the request and that declaration
 * disagree in either direction. It is behavioural rather than textual: it reads
 * the payload the page actually sends, so a key added anywhere in a builder
 * shows up whether or not anyone remembered this file.
 */

/** How one request key relates to the page's link contract. */
export type RequestKeyBinding =
  /** The contract fields this key is derived from. */
  | { from: readonly string[] }
  /** Why a shared view cannot carry this key. */
  | { notShareable: string };

export interface RequestCensus {
  /** The page's route, for messages. */
  page: string;
  /** Every key in the request payload, classified. */
  request: Record<string, RequestKeyBinding>;
  /**
   * Contract fields the request does not carry, with the reason. These are
   * page-only controls: a view can carry them without the API ever seeing them.
   */
  viewOnly?: Record<string, string>;
}

/**
 * Compare a page's declared census with the request it really sends.
 *
 * `request` must be a payload that exercises every optional key, or the
 * "classified but never sent" check reports a key that is merely conditional.
 * Returns the problems found, so a test can assert the list is empty and print
 * what is wrong when it is not.
 */
export function auditRequestCensus(
  fieldKeys: readonly string[],
  census: RequestCensus,
  request: Record<string, unknown>,
): string[] {
  const problems: string[] = [];
  const fields = new Set(fieldKeys);

  for (const key of Object.keys(request)) {
    if (!(key in census.request)) {
      problems.push(
        `${census.page}: request key "${key}" is not classified — carry it in a link field or say why it cannot be shared`,
      );
    }
  }

  for (const [key, binding] of Object.entries(census.request)) {
    if (!(key in request)) {
      problems.push(`${census.page}: request key "${key}" is classified but never sent`);
    }
    if (!('from' in binding)) {
      if (binding.notShareable.trim() === '') {
        problems.push(`${census.page}: request key "${key}" gives no reason it cannot be shared`);
      }
      continue;
    }
    if (binding.from.length === 0) {
      problems.push(`${census.page}: request key "${key}" derives from no field`);
    }
    for (const field of binding.from) {
      if (!fields.has(field)) {
        problems.push(`${census.page}: request key "${key}" derives from unknown field "${field}"`);
      }
    }
  }

  const carried = new Set<string>();
  for (const binding of Object.values(census.request)) {
    if ('from' in binding) for (const field of binding.from) carried.add(field);
  }

  for (const key of Object.keys(census.viewOnly ?? {})) {
    if (!fields.has(key)) {
      problems.push(`${census.page}: view-only field "${key}" is not in the contract`);
    }
    if (carried.has(key)) {
      problems.push(`${census.page}: field "${key}" is both carried and view-only`);
    }
    if ((census.viewOnly?.[key] ?? '').trim() === '') {
      problems.push(`${census.page}: view-only field "${key}" gives no reason`);
    }
  }

  for (const key of fieldKeys) {
    if (carried.has(key)) continue;
    if (census.viewOnly && key in census.viewOnly) continue;
    problems.push(
      `${census.page}: contract field "${key}" is not carried by the request — bind it or list it as view-only`,
    );
  }

  return problems;
}
