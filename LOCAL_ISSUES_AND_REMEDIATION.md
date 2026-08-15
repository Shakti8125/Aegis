# Aegis — Local Issues & Remediation Logic

Audit of architecture and UI rendering issues relevant to **local development and testing**. Public deployment concerns (HTTPS/WSS, Vercel/Render, cloud secrets, production CORS) are intentionally omitted — handle those separately.

Sources: `PLAN.md`, `ARCHITECTURE.md`, `PHASE_8_REPORT.md`, `challenges_faced.md`, and cross-check against current backend/frontend code.

---

## How to read this document

Each issue includes:

| Field | Meaning |
|-------|---------|
| **Severity** | Impact during local dev / demo recording |
| **Observed in** | Files or behavior |
| **Problem** | What goes wrong |
| **Correction logic** | How to fix it (design + steps), without prescribing a full patch |

---

## 1. UI rendering

### 1.1 D3 graph flickers and resets every tick (confirmed)

**Severity:** High  
**Observed in:** `frontend/src/components/ClusterGraph.tsx`

**Problem:**  
The force-directed graph is torn down and rebuilt on every WebSocket `tick` frame. The `useEffect` that owns D3 depends on `nodesData` / `linksData`, which get new references whenever `cluster` updates. Inside the effect:

- `svg.selectAll("*").remove()` wipes the entire SVG
- A new `d3.forceSimulation` starts from scratch
- Zoom/pan transform is lost
- Pulse rings and edge-trace animations are interrupted mid-flight

This directly conflicts with `PLAN.md` §9: nodes should pulse briefly and cited edges should trace — motion requires **stable DOM nodes**, not a full remount every ~100ms.

**Correction logic:**

1. **Split “structure” from “telemetry” updates**
   - Run force simulation setup **once** when service IDs or edge topology change (nodes added/removed, edges added/removed).
   - On each tick, update only **attributes**: circle stroke/fill, label text, edge stroke color/width/dasharray, pulse overlay visibility.

2. **Stable node identity**
   - Key D3 joins on `d.id` (service id), not array index.
   - Use `selection.data(nodes, (d) => d.id)` enter/update/exit pattern.

3. **Preserve zoom**
   - Store zoom transform in a `useRef`; re-apply to the root `<g>` after structural changes only, not every tick.

4. **Pulse as overlay, not remount trigger**
   - Keep pulse ring elements in the DOM; toggle opacity/class via a separate effect keyed on `actions[0]`, not on full graph rebuild.

5. **Throttle simulation alpha**
   - After initial layout, set `simulation.alphaTarget(0)` and only bump alpha when topology changes — not on health metric updates.

6. **Acceptance check:** Start sim, zoom into a node, let 50+ ticks pass — layout and zoom stay stable; pulse and edge highlight animate without flicker.

---

### 1.2 Mock “healthy” cluster shown while WebSocket is disconnected

**Severity:** High (operator trust)  
**Observed in:** `frontend/src/context/WebSocketContext.tsx` (`MOCK_CLUSTER`), `frontend/src/components/Header.tsx`

**Problem:**  
Initial state is a fully healthy 12-service mock graph. Header shows tick `0`, mean health `100%`, SLA `0%` even when the status badge reads `disconnected` or `error`. Users cannot tell real telemetry from placeholder data.

**Correction logic:**

1. **Connection-aware initial state**
   - Initialize `cluster` as `null` (or an explicit `EmptyCluster` sentinel), not `MOCK_CLUSTER`.

2. **Render gates**
   - `ClusterGraph`: show skeleton / “Waiting for connection…” when `cluster === null` or `status !== "connected"`.
   - Header metrics: show `—` or hide values until first real `tick` frame arrives.

3. **Optional dev-only mock**
   - If offline UI dev is needed, gate mock data behind `import.meta.env.DEV && import.meta.env.VITE_USE_MOCK === "true"` — never default in the live provider.

4. **Acceptance check:** Stop backend → dashboard shows empty/disconnected state, not a green cluster.

---

### 1.3 Incident Feed always displays “LIVE”

**Severity:** Medium  
**Observed in:** `frontend/src/components/IncidentFeed.tsx`

**Problem:**  
The pulsing green “LIVE” badge renders unconditionally. Stale actions from a prior session or mock data can appear “live” while WebSocket is disconnected.

**Correction logic:**

1. Read `status` from `useWebSocket()` in `IncidentFeed`.
2. Render badge by state:
   - `connected` → “LIVE” (green pulse)
   - `connecting` → “CONNECTING” (amber)
   - `disconnected` / `error` → “OFFLINE” (gray/red, no pulse)
3. Optionally dim or freeze the action list when not connected.
4. Clear `actions` on `onclose` / `onerror` so stale entries do not linger.

---

### 1.4 Wrong dependency edge highlighted for narrator trace

**Severity:** Medium (causal misrepresentation)  
**Observed in:** `frontend/src/context/WebSocketContext.tsx`, `backend/ws.py`, `backend/models.py`

**Problem:**  
`PLAN.md` §9 and `ARCHITECTURE.md` say the **specific dependency edge cited by the narrator** should animate. Current logic picks the first edge where `target_service` matches **either** source or target — often not the causal edge.

**Correction logic:**

1. **Backend: attach cited edge to `ActionEvent`**
   - In `_build_action_context` / narrator pipeline, the narrator already knows `dependencies` and `dependents`.
   - Add optional fields to `ActionEvent`, e.g. `cited_edge_source` and `cited_edge_target` (or `cited_edges: list[{source, target}]`).
   - Rule-based narrator: pick the dependency with worst `error_rate` or highest `p99_latency_ms` among facts passed in.
   - LLM narrator: parse structured output (e.g. `EDGE: svc-03 -> svc-08`) or pass edge IDs in the prompt and require citing one from the provided list.

2. **Frontend: use explicit edge**
   - Replace heuristic `find` on `cluster.edges` with `setHighlightedEdge({ source: act.cited_edge_source, target: act.cited_edge_target })`.

3. **Fallback:** If no cited edge, skip highlight rather than guess.

4. **Acceptance check:** On a `restart` of `svc-03` with upstream caller `svc-01`, the animated edge matches the narration text, not an arbitrary adjacent edge.

---

### 1.5 Color-only health encoding (accessibility)

**Severity:** Low  
**Observed in:** `PLAN.md` §9 token system, `ClusterGraph.tsx`, `NodeInspector.tsx`, `Header.tsx`

**Problem:**  
Health is communicated only via green/amber/red. Colorblind users and grayscale recordings may misread state.

**Correction logic:**

1. Add non-color cues already partially present in `NodeInspector` (text badges) to the graph nodes: e.g. small icon or stroke pattern (solid / dashed / dotted) per health band.
2. Keep `#3DDC97` / `#F5A623` / `#E5484D` tokens but pair each with a **text label on hover/selection** (“Healthy”, “Degraded”, “Critical”).
3. Document the pairing in `PLAN.md` §9 when updating docs later.

---

## 2. WebSocket & backend (local)

### 2.1 WebSocket reconnect loop was fixed; dependency on hardcoded URL remains

**Severity:** Medium (local ergonomics)  
**Observed in:** `challenges_faced.md` §3 vs §4, `WebSocketContext.tsx`, `vite.config.ts`

**Problem:**  
`challenges_faced.md` contains **contradictory** resolutions:

- §3: connect via Vite proxy (`window.location.host`)
- §4: connect directly to `ws://127.0.0.1:8000`

Current code uses hardcoded `ws://127.0.0.1:8000/ws/live`. That works when backend is on 8000, but bypasses the Vite proxy defined in `vite.config.ts` and breaks if you change ports or run frontend-only against a different host.

**Correction logic (local only):**

1. Pick **one** strategy and document it in `challenges_faced.md`:
   - **Recommended for dev:** proxy path  
     `const wsUrl = \`${protocol}//${window.location.host}/ws/live\``  
     with Vite proxy target `ws://127.0.0.1:8000`.
   - Ensures same-origin, avoids Windows IPv6 `localhost` vs `127.0.0.1` issues described in the challenges doc.

2. Optional: `import.meta.env.VITE_WS_URL` override for ad-hoc local setups — still not “production deployment,” just flexible localhost.

3. Remove or mark superseded the conflicting section in `challenges_faced.md` when you next edit docs.

---

### 2.2 Unvalidated WebSocket `start` command → crashes or resource exhaustion

**Severity:** Medium (local DoS — e.g. runaway sim in a tab)  
**Observed in:** `backend/main.py`, `backend/ws.py`, `marl/vec_env.py`

**Problem:**  
Client sends arbitrary JSON:

- Unknown `scenario` → `KeyError` in `scenario_overrides()` inside the async sim task → error broadcast, stack trace in logs.
- Huge `max_cycles` (e.g. 999999) or `tick_delay_ms: 0` → CPU spin locally.
- No schema validation on message shape.

**Correction logic:**

1. **Validate before spawning `SimulationRunner`**
   - `scenario`: must be in `SCENARIOS.keys()`; else send `WsFrame(type=ERROR, message="unknown scenario")` and do not start task.
   - `max_cycles`: clamp to e.g. `[1, 500]` (match training/eval expectations).
   - `tick_delay_ms`: clamp to e.g. `[20, 2000]`.
   - `seed`: coerce to `int`, reject non-numeric.

2. **Use Pydantic model** for inbound WS commands (`WsStartCommand`) — same pattern as outbound `WsFrame`.

3. **Wrap sim task** with try/except that converts `KeyError` / `ValueError` to client-visible `ERROR` frames without `logger.exception` noise for expected bad input.

4. **Acceptance check:** Send `{"command":"start","scenario":"bogus"}` → ERROR frame, no task spawned, server stays healthy.

---

### 2.3 Global broadcast — one tab’s simulation drives all tabs

**Severity:** Medium (local multi-tab confusion)  
**Observed in:** `backend/ws.py` (`ConnectionManager.broadcast`), `backend/main.py`

**Problem:**  
When one browser tab sends `start`, tick frames go to **every** connected WebSocket. Opening two tabs locally causes both to show the same sim; stopping in one tab may confuse the other.

**Correction logic:**

1. **Per-connection sim tasks (minimal fix)**
   - Already keyed by `ws_id` for task storage — change `broadcast(frame)` to `send(ws, frame)` for tick frames originating from that client’s runner.
   - Optionally broadcast only `episode_end` or system messages if desired.

2. **Alternative:** single global sim with explicit “driver” client — document that only one tab should control; worse UX, not recommended.

3. **Acceptance check:** Two tabs open; start sim in tab A only → tab B does not update until it sends its own `start`.

---

### 2.4 Auto-reconnect without backoff

**Severity:** Low  
**Observed in:** `frontend/src/context/WebSocketContext.tsx` (`setTimeout(connect, 3000)` on close)

**Problem:**  
If backend is down, client reconnects every 3s forever — noisy console, wasted local resources.

**Correction logic:**

1. Exponential backoff with cap (e.g. 3s → 6s → 12s → max 30s).
2. Reset backoff on successful `onopen`.
3. Optional: pause reconnect when `document.hidden` (tab in background).

---

## 3. LLM ops layer (local behavior vs docs)

### 3.1 Live WebSocket path uses stub LLM, not real narrator

**Severity:** Medium (demo fidelity)  
**Observed in:** `backend/main.py` (`make_client("stub", response="")`), `PHASE_8_REPORT.md` Phase 5 “100% compliant”

**Problem:**  
`/ws/live` simulations use `Narrator` with a stub client — narrations come from **templates**, not Ollama/Gemini. Local testing of “grounded LLM narration” via the dashboard does not exercise the real ops layer unless you change backend wiring.

**Correction logic:**

1. Use `make_client()` with env-driven backend (same as `demo/e2e_runner` or CLI):
   - Default: Ollama if `OLLAMA_HOST` reachable.
   - Fallback: Gemini if `GEMINI_API_KEY` set.
   - Fallback: rule-based templates if neither available (log once at startup).

2. Do **not** hardcode `"stub"` in `ws_live` — stub only for unit tests.

3. **Acceptance check:** Start Ollama locally, run sim → incident feed shows LLM-style varied prose grounded in passed facts; stop Ollama → graceful template fallback with visible log warning.

---

### 3.2 “Zero-hallucination” is overstated — no verification step

**Severity:** Medium (trust / audit)  
**Observed in:** `ARCHITECTURE.md` §4, `PLAN.md` Phase 5, `ops_layer/narrator.py`

**Problem:**  
Docs claim narrations cite only graph facts. Enforcement is prompt-only. The model can still invent causes; nothing validates output against `ActionContext` before sending to the UI.

**Correction logic:**

1. **Structured narrator output**
   - Ask LLM to return JSON: `{ "text": "...", "cited_facts": ["svc-03 error_rate=0.12", "edge svc-01->svc-03"] }`.
   - Render only `text`; store `cited_facts` for audit panel.

2. **Lightweight post-check (deterministic)**
   - Every service id mentioned in `text` must appear in `ActionContext` (target, dependencies, dependents, faults).
   - Reject/regenerate or fall back to template if unknown ids detected (regex on `svc-\d+` pattern).

3. **UI:** Optionally show “Sources” expandable list from `cited_facts` — aligns with `PLAN.md` §9 causal trace goal.

4. Update marketing language in docs from “zero-hallucination” to **“prompt-grounded with optional fact validation”** when you revise docs later.

---

### 3.3 Safety supervisor fails open on LLM errors

**Severity:** Medium (safety story vs behavior)  
**Observed in:** `ops_layer/safety_supervisor.py` (`_llm_check` → `LLMError` → `return None` → ALLOW)

**Problem:**  
When LLM semantic policy check fails (timeout, API down), proposed actions are **allowed**. Docs present the supervisor as an enterprise veto layer; fail-open is reasonable for local dev but must be explicit and configurable.

**Correction logic:**

1. Add `on_llm_failure: "allow" | "veto" | "no_op"` to `SafetySupervisor` config.
2. **Local dev default:** `"allow"` with logged warning (current behavior, but visible).
3. **Demo recording default:** `"no_op"` — force no-op on LLM failure so unsafe actions do not slip through when the “smart” layer is down.
4. Rule-based policies (`DEFAULT_POLICIES`) always run first regardless — document that LLM veto is additive, not sole gate.

---

## 4. Phase 8 local demo (`kind` / kubectl)

### 4.1 Kubectl adapter: unsanitized resource names

**Severity:** High **if** you run real `kind` demo with `dry_run=False` locally  
**Observed in:** `demo/kubectl_adapter.py`, `ARCHITECTURE.md` Layer 8

**Problem:**  
`target_name` is interpolated into kubectl arguments without validation. Malformed or injected names (spaces, `;`, shell metacharacters if ever passed through shell) could cause unexpected commands. Currently uses `subprocess.run` with list args (good — no shell), but invalid K8s names still fail opaquely or could target wrong resources if names are wrong.

**Correction logic:**

1. Validate `target_name` against Kubernetes naming regex: `^[a-z0-9]([-a-z0-9]*[a-z0-9])?$` (for DNS-1123 subdomain).
2. Reject before subprocess if invalid; return structured error dict.
3. Keep `dry_run=True` as default in `e2e_runner` until validation exists.
4. Map simulator service ids (`svc-03`) to deployment names via an explicit **allowlist dict** — never pass raw agent ids straight to kubectl.

---

### 4.2 RL actions → real cluster without human confirmation gate

**Severity:** Medium (local kind safety)  
**Observed in:** `PLAN.md` Phase 8, `demo/e2e_runner.py`

**Problem:**  
Architecture describes automatic execution of restart/scale/isolate on a real cluster. Even locally, one mis-map or bad policy can disrupt the kind cluster mid-recording.

**Correction logic:**

1. **Two-phase demo mode:** `propose` → log intended kubectl → `confirm` flag or stdin prompt before execute.
2. **`--dry-run` default** on CLI; require explicit `--execute` for real mutations.
3. Run only in dedicated `aegis-demo` namespace with RBAC limited to that namespace (document in demo README when you add it).

---

## 5. Documentation inconsistencies (local dev confusion)

These do not break code by themselves but cause wasted debugging time — align when you next edit docs.

| Issue | Where | Correction logic |
|-------|-------|------------------|
| Vite dev port `5173` vs `3000` | `PHASE_8_REPORT.md` says 5173; `vite.config.ts` uses 3000 | Pick one port; update all docs to match `vite.config.ts` |
| Conflicting WebSocket connection fix | `challenges_faced.md` §3 vs §4 | Keep proxy-based approach; mark §4 direct-IP as superseded |
| Phase 5 “100% LLM compliant” vs stub in WS | `PHASE_8_REPORT.md` vs `backend/main.py` | Note WS path uses stub unless env client wired |
| Signature edge trace vs heuristic | `PLAN.md` §9 vs `WebSocketContext.tsx` | Document need for `cited_edge_*` on `ActionEvent` until implemented |

---

## 6. Suggested fix order (local testing)

Work in this order for maximum impact while you test locally:

1. **Graph flicker (§1.1)** — you already observed this; biggest UX win.
2. **Disconnected / mock state (§1.2, §1.3)** — restores trust in the dashboard.
3. **WS input validation (§2.2)** — prevents accidental local runaway sims.
4. **Per-tab sim isolation (§2.3)** — if you use multiple tabs.
5. **Cited edge in `ActionEvent` (§1.4)** — matches PLAN §9 signature behavior.
6. **Real LLM client on WS path (§3.1)** — when testing narration quality locally.
7. **Kubectl validation (§4.1–4.2)** — before any non-dry-run kind demo.

---

## 7. Out of scope for this document

The following were identified in the broader audit but deferred per your request:

- HTTPS / WSS and mixed-content rules for hosted frontend
- Vercel, Render, Fly.io, Neo4j Aura deployment hardening
- Production CORS, authn/z, rate limiting on public endpoints
- Cloud API key exposure and rotation
- Public exposure of `/api/metrics` and training artifacts

Track those in a separate deployment hardening doc when you are ready.

---

## 8. Quick reference — files to touch when implementing

| Issue | Primary files |
|-------|----------------|
| Graph flicker | `frontend/src/components/ClusterGraph.tsx` |
| Mock / LIVE state | `frontend/src/context/WebSocketContext.tsx`, `IncidentFeed.tsx`, `Header.tsx` |
| Edge trace | `backend/models.py`, `backend/ws.py`, `ops_layer/narrator.py`, `WebSocketContext.tsx` |
| WS validation | `backend/main.py`, new `backend/ws_commands.py` (optional) |
| Per-client broadcast | `backend/main.py`, `backend/ws.py` |
| LLM on live path | `backend/main.py`, `ops_layer/llm_client.py` |
| Narration verification | `ops_layer/narrator.py` |
| Supervisor fail mode | `ops_layer/safety_supervisor.py` |
| Kubectl safety | `demo/kubectl_adapter.py`, `demo/e2e_runner.py` |

No code changes are included here — this file is the remediation spec only.
