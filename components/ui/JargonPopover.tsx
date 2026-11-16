"use client";

import React from "react";
import { Popover } from "@patternfly/react-core";
import { HelpIcon } from "@patternfly/react-icons";
import { findGlossaryEntry } from "@/lib/help/glossary";

interface JargonPopoverProps {
  term: string;
  /** Page-specific explanation. Used when the term is not in the shared glossary. */
  explanation?: string;
  /** Portal target for use inside clipped or transformed containers such as flip cards. */
  appendTo?: HTMLElement | ((ref?: HTMLElement) => HTMLElement) | "inline";
}

export function JargonPopover({ term, explanation, appendTo }: JargonPopoverProps) {
  const entry = findGlossaryEntry(term);
  const body = entry?.body ?? explanation;
  if (!body) return null;

  return (
    <Popover
      aria-label={`${term} explanation`}
      headerContent={<div className="gc-label">{entry?.title ?? term}</div>}
      bodyContent={<div className="gc-body">{body}</div>}
      position="auto"
      maxWidth="360px"
      appendTo={appendTo}
    >
      <button
        type="button"
        aria-label={`Help for ${term}`}
        style={{
          background: "none",
          border: "none",
          padding: "0 4px",
          cursor: "pointer",
          color: "var(--gc-text-2)",
          verticalAlign: "middle",
          display: "inline-flex",
          alignItems: "center",
        }}
      >
        <HelpIcon style={{ fontSize: "12px" }} />
      </button>
    </Popover>
  );
}

