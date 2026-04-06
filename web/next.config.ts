/**
 * Next.js config for the diabetes demo front-end (defaults only; extend as needed).
 */
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Required for the production Dockerfile (multi-stage copy of .next/standalone).
  output: "standalone",
};

export default nextConfig;
