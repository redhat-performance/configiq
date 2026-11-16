import { NextResponse } from 'next/server';
import { z } from 'zod';
import { recordHelpTour } from '@/lib/otel-metrics';

export const dynamic = 'force-dynamic';

const eventSchema = z.object({
  id: z.string().min(1).max(64),
  phase: z.enum(['started', 'completed']),
});

/**
 * Records a tour start/complete event. Deliberately fire-and-forget: the
 * client ignores the response, and an unknown tour id is accepted but not
 * counted (see `recordHelpTour`), so a stale bundle cannot add metric labels.
 */
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

  recordHelpTour(parsed.data.id, parsed.data.phase);
  return NextResponse.json({ ok: true });
}
