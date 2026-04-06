/**
 * BFF route: forwards JSON body to the Python ML service POST /predict.
 * Proxies status codes; logs request id and latency. ML_SERVICE_URL defaults to localhost:8000.
 */
import { NextRequest, NextResponse } from "next/server";

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";

/** POST handler — returns ML JSON or 502 if the backend is unreachable. */
export async function POST(request: NextRequest) {
  const requestId = crypto.randomUUID();
  const start = Date.now();
  try {
    const body = await request.json();
    const res = await fetch(`${ML_SERVICE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-request-id": requestId },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    const elapsed = Date.now() - start;
    console.info(
      JSON.stringify({
        request_id: requestId,
        path: "/api/predict",
        status: res.status,
        elapsed_ms: elapsed,
      })
    );
    if (!res.ok) {
      return NextResponse.json(
        { detail: data.detail || "ML service error" },
        { status: res.status }
      );
    }
    return NextResponse.json(data);
  } catch (err) {
    const elapsed = Date.now() - start;
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error(
      JSON.stringify({
        request_id: requestId,
        path: "/api/predict",
        status: 502,
        elapsed_ms: elapsed,
        error: message,
      })
    );
    return NextResponse.json(
      { detail: `Failed to call ML service: ${message}` },
      { status: 502 }
    );
  }
}
