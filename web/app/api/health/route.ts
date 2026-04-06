/**
 * Aggregated health: proxies GET /health on the ML service; 503 if down or unreachable.
 */
import { NextResponse } from "next/server";

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";

/** GET handler — includes ml_service payload when healthy. */
export async function GET() {
  const start = Date.now();
  try {
    const res = await fetch(`${ML_SERVICE_URL}/health`, {
      cache: "no-store",
    });
    const data = await res.json();
    const elapsed = Date.now() - start;
    console.info(
      JSON.stringify({
        path: "/api/health",
        status: res.ok ? 200 : 503,
        elapsed_ms: elapsed,
      })
    );
    if (!res.ok) {
      return NextResponse.json(
        { status: "unhealthy", ml_service: data },
        { status: 503 }
      );
    }
    return NextResponse.json({ status: "ok", ml_service: data });
  } catch {
    const elapsed = Date.now() - start;
    console.error(
      JSON.stringify({
        path: "/api/health",
        status: 503,
        elapsed_ms: elapsed,
        error: "ml_service_unreachable",
      })
    );
    return NextResponse.json(
      { status: "unhealthy", ml_service: "unreachable" },
      { status: 503 }
    );
  }
}
