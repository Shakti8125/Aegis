// Phase 7 — owned by the frontend-builder subagent.
// The token system (health color scale, type pairing) is specified in
// PLAN.md §9 and belongs here once the dashboard build starts.

import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
} satisfies Config;
