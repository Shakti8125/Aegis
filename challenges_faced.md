# Challenges Faced & Solutions

This document tracks technical challenges, warnings, and error resolutions encountered during the development and maintenance of the Aegis frontend and backend services.

---

## 1. Vite 8 Config & Bundler Deprecations (`jsx` Invalid Key & `__dirname` Warning)

### Issues Encountered
When running `npm run dev` in `frontend/`, Vite outputted several warnings and invalid option errors:
- `(!) Your Vite config uses features that are unsupported by configLoader: 'native' ... __dirname (vite.config.ts). Use import.meta.dirname instead.`
- `[vite] warning: esbuild option was specified by "vite:react-babel" plugin. This option is deprecated, please use oxc instead.`
- `Warning: Invalid input options (1 issue found) - For the "jsx". Invalid key: Expected never but received "jsx".`
- `npm error ERESOLVE could not resolve peer dependency vite@"^4 || ^5 || ^6 || ^7"` when attempting to update plugins under Vite 8.

### Root Cause
1. **Vite Native Config Loader**: Node.js ES Modules deprecate Node's legacy `__dirname` global in favor of `import.meta.dirname`.
2. **Vite 8 & Rolldown Incompatibility**: Vite 8 uses Rolldown as its underlying bundler/optimizer. The Babel-based `@vitejs/plugin-react` v4 passes `esbuild` and top-level `jsx` options that Rolldown marks as deprecated or invalid input options.
3. **Peer Dependency Mismatches**: Vite 8 is a major pre-release version with strict peer dependency bounds in existing Vite plugins.

### Resolution
- Updated [vite.config.ts](file:///c:/Users/Shakti/Documents/Aegis/frontend/vite.config.ts) to replace `path.resolve(__dirname, "./src")` with `path.resolve(import.meta.dirname, "./src")`.
- Updated [package.json](file:///c:/Users/Shakti/Documents/Aegis/frontend/package.json) to use stable Vite `^6.2.0` and `@vitejs/plugin-react` `^4.3.4`.
- Verified `npm run build` executes cleanly with 0 errors and 0 deprecation warnings in 3.8s.

---

## 2. Frontend Disconnected WebSocket Connection Loop

### Issue Encountered
The Aegis frontend dashboard header rendered the live WebSocket status badge as **`Disconnected`** (red dot) even while the FastAPI backend server on `http://localhost:8000/` was running and listening on port 8000.

### Root Cause
In [WebSocketContext.tsx](file:///c:/Users/Shakti/Documents/Aegis/frontend/src/context/WebSocketContext.tsx), the `connect` callback was wrapped in `useCallback` with `[cluster?.edges]` listed in its dependency array:

```tsx
const connect = useCallback(() => {
  ...
  const edgeMatch = cluster?.edges.find((e) => e.source === src || e.target === src);
  ...
}, [cluster?.edges]);

useEffect(() => {
  connect();
  return () => {
    if (wsRef.current) wsRef.current.close();
  };
}, [connect]);
```

Every time a `tick` frame arrived or state updated, `setCluster(frame.cluster)` updated `cluster`, producing a new array reference for `cluster?.edges`. This recreated the `connect` function, which triggered `useEffect`'s cleanup function (`wsRef.current.close()`). The active WebSocket connection was forcibly closed on every tick, setting `status` to `"disconnected"` and causing a constant reconnect loop.

### Resolution
- Created a `clusterRef` using `useRef<ClusterSnapshot | null>(cluster)` to store the latest cluster snapshot.
- Updated `onmessage` in `WebSocketContext.tsx` to read `clusterRef.current?.edges` instead of directly closing over `cluster?.edges`.
- Changed `connect`'s `useCallback` dependency list to `[]` (empty array), preventing unnecessary socket closures and maintaining a persistent connection to `ws://localhost:8000/ws/live`.

---

## 3. Windows IPv6 `localhost` Resolution & Vite Proxy Integration

### Issue Encountered
On Windows, direct cross-port connections using `ws://${window.location.hostname}:8000/ws/live` can fail with `ERR_CONNECTION_REFUSED` if the browser resolves `localhost` to `[::1]` (IPv6) while the backend Python server (`uvicorn`) is bound to `127.0.0.1` (IPv4).

### Resolution
Updated [WebSocketContext.tsx](file:///c:/Users/Shakti/Documents/Aegis/frontend/src/context/WebSocketContext.tsx) to connect through the Vite dev server proxy using:
```ts
const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
const wsUrl = `${protocol}//${window.location.host}/ws/live`;
```
Because [vite.config.ts](file:///c:/Users/Shakti/Documents/Aegis/frontend/vite.config.ts) proxies `/ws` directly to `ws://127.0.0.1:8000`, this eliminates IPv4/IPv6 address mismatches and cross-origin WebSocket restrictions on Windows.

---

## 4. Vite Dev Server WebSocket Proxy Failure on Windows (`ws://localhost:3000/ws/live`)

### Issue Encountered
Chrome DevTools console logged red errors:
`WebSocket connection to 'ws://localhost:3000/ws/live' failed`
when attempting to proxy WebSocket upgrade requests through Vite's dev server on `port 3000`.

### Root Cause
1. **IPv6 Resolution in Proxy Target**: Vite's proxy target `target: "ws://localhost:8000"` attempted to connect to `[::1]:8000` (IPv6), while the Python `uvicorn` backend was listening on `127.0.0.1:8000` (IPv4).
2. **WebSocket Upgrade Failures**: Proxying WebSocket upgrades through Vite on Windows fails if the proxy target uses `localhost` instead of explicit IPv4 `127.0.0.1`.

### Resolution
- Updated [vite.config.ts](file:///c:/Users/Shakti/Documents/Aegis/frontend/vite.config.ts) proxy targets to explicit IPv4 (`http://127.0.0.1:8000` and `ws://127.0.0.1:8000`).
- Updated [WebSocketContext.tsx](file:///c:/Users/Shakti/Documents/Aegis/frontend/src/context/WebSocketContext.tsx) to target `ws://127.0.0.1:8000/ws/live` directly, ensuring an instant, reliable connection across all browsers.

---

## 5. Stale Backend Process & Host Interface Binding (`0.0.0.0` vs `127.0.0.1`)

### Issue Encountered
DevTools Sources tab hit an exception on line 101 (`new WebSocket(wsUrl)`), indicating the WebSocket connection attempt was refused by port 8000.

### Root Cause
1. **Stale Process with Exhausted Connections**: The previous backend process (PID 40788) had accumulated dozens of orphaned sockets in `TIME_WAIT` state from repeated dev reloads.
2. **Interface Binding Restriction**: Running `uvicorn` bound strictly to `127.0.0.1` rejected dual-stack IPv4/IPv6 client connections.

### Resolution
- Terminated the stale process (PID 40788).
- Restarted the Aegis FastAPI backend bound to `0.0.0.0:8000` (`python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`).
- Verified real-time WebSocket connection and frame streaming using an automated test script (`test_ws_conn.py`).

---

## 6. Missing Uvicorn WebSocket Engine (`uvicorn[standard]` & `httptools` / `websockets`)

### Issue Encountered
DevTools console logged:
`WebSocket connection to 'ws://localhost:3000/ws/live' failed: WebSocket is closed before the connection is established.`
The Uvicorn backend log revealed:
```text
WARNING: Unsupported upgrade request.
WARNING: No supported WebSocket library detected. Please use "pip install 'uvicorn[standard]'", or install 'websockets' or 'wsproto' manually.
INFO: 127.0.0.1 - "GET /ws/live HTTP/1.1" 404 Not Found
```

### Root Cause
FastAPI and Uvicorn require an explicit WebSocket engine (`websockets`, `wsproto`, or `httptools` from `uvicorn[standard]`) to process HTTP `Upgrade: websocket` requests. When the virtual environment lacked `uvicorn[standard]`, Uvicorn rejected all WebSocket upgrade requests with **HTTP 404 Not Found**.

### Resolution
1. Installed `httptools`, `watchfiles`, `websockets-17.0.1`, and `uvicorn[standard]` into `.venv` (`pip install websockets "uvicorn[standard]"`).
2. Restarted Uvicorn using `.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`.
3. Confirmed WebSocket Handshake `[accepted]` and verified tick streaming continuously.




