import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function GET() {
  const gatewayUrl = process.env.AISIMULATORS_GATEWAY_URL;

  // Require explicit gateway URL configuration
  if (!gatewayUrl) {
    return NextResponse.json(
      {
        status: 'unhealthy',
        message: 'Gateway configuration missing',
      },
      { status: 200 }
    );
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5s timeout

    // Check if AISimulators API is reachable via gateway
    const apiResponse = await fetch(`${gatewayUrl}/systems`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      signal: controller.signal,
      cache: 'no-store',
    });

    clearTimeout(timeoutId);

    // Release response body to allow connection reuse
    if (!apiResponse.ok) {
      await apiResponse.body?.cancel();
      return NextResponse.json(
        {
          status: 'degraded',
          message: 'API connectivity issue',
        },
        { status: 200 }
      );
    }

    // Consume body to release connection
    await apiResponse.json();

    return NextResponse.json(
      {
        status: 'healthy',
        message: 'ConfigIQ webapp is running',
        timestamp: new Date().toISOString(),
      },
      { status: 200 }
    );
  } catch (error) {
    return NextResponse.json(
      {
        status: 'unhealthy',
        message: 'Health check failed',
      },
      { status: 200 }
    );
  }
}
