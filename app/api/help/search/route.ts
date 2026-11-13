import { NextResponse } from 'next/server';
import { z } from 'zod';
import { recordHelpSearch } from '@/lib/otel-metrics';

export const dynamic = 'force-dynamic';

/**
 * Counts a help search. The query text is not accepted, let alone logged:
 * knowing whether the index answered is enough to spot documentation gaps
 * without recording what an operator typed.
 */
const eventSchema = z.object({
  matched: z.number().int().min(0).max(10_000),
});

export async function POST(request: Request) {
  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ ok: false }, { status: 400 });
  }

  const parsed = eventSchema.safeParse(payload);
  if (!parsed.success) {
    return NextResponse.json({ ok: false }, { status: 400 });
  }

  recordHelpSearch(parsed.data.matched);
  return NextResponse.json({ ok: true });
}
