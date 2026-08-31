# Deploying the live demo to the internet

`pi/tools/demo_explore.py` — the simulated-room exploration demo, both the
2D map and the 3D scene — deployed on [Render](https://render.com)'s free
tier, chosen after checking current (not remembered) free-tier terms: no
credit card, a real Python web service (not tied to a specific framework),
and 512MB/0.1 CPU free forever, not a trial. Railway and Fly.io have both
moved away from genuinely free, no-card tiers; PythonAnywhere's free tier is
built around WSGI apps and doesn't fit this project's plain `http.server`
approach as cleanly. Re-check current terms before deploying, since these
change — this isn't a permanent guarantee.

GitHub Pages was considered and deliberately not used: it only serves static
files and cannot run the Python backend this demo's mission/simulation logic
depends on. Making that work would mean porting the simulator, mapping, and
exploration logic to JavaScript — a real, separate undertaking, not a deploy
step. If that ever happens, it belongs on its own branch, same as the
camera-based 3D room mapping work.

## Deploy steps

1. Push this repo to GitHub (already done — `origin/main`).
2. In Render: **New > Blueprint**, connect the GitHub repo. Render
   auto-detects `render.yaml` at the repo root and proposes the `scout-demo`
   service from it — review and deploy.
3. First load after a period of inactivity takes 30-60 seconds (free-tier
   services sleep after ~15 minutes with no HTTP traffic, and a visit wakes
   them from a cold start). Once awake, both pages are live:
   - `https://<your-service>.onrender.com/` — the 2D map
   - `https://<your-service>.onrender.com/scene.html` — the 3D scene
4. Because free-tier services actually stop (not just pause) after 15
   minutes idle, the whole process — including the background simulation
   thread — restarts from scratch on the next visit. `SCOUT_LOOP=1` (set in
   render.yaml) handles the *other* case: a visitor who stays on the page
   long enough to watch a lap finish sees a fresh lap start automatically,
   rather than the demo going still once `DONE`.

## Security review

The question this section actually answers: **can a visitor to the public
URL do anything other than watch?** No — and that's by construction, not by
add-on hardening. Verified in `pi/tests/test_server.py`, not just asserted:

- **The entire HTTP surface is three fixed pages and two read-only JSON
  endpoints**, all GET-only. `_STATIC_PAGES` maps a closed, fixed set of
  paths to fixed filenames — a request's path is looked up in that dict, and
  an unrecognized path is a 404 before any filesystem access happens. There
  is nowhere a request's own path or content is interpolated into a file
  path, a command, a shell call, or a database query (there's no database).
  Classic traversal payloads (`/../../etc/passwd`, `/etc/passwd`) are
  confirmed to 404 like any other unrecognized path, not to read anything.
- **No command/control endpoint is exposed over HTTP at all.** The mission
  that drives the simulated robot runs entirely server-side, in Python,
  driven by the mission loop — not by anything a request can trigger. A
  visitor can watch the same shared simulation every other visitor sees;
  nothing they send changes it. POST (or any method besides GET) gets
  `BaseHTTPRequestHandler`'s built-in "unsupported method" response, since no
  handler for it is defined anywhere.
- **No secrets, credentials, or user data exist anywhere in this path.**
  Nothing is collected, logged, or stored about visitors.
- **Security headers on every response**: `X-Frame-Options: DENY` (the page
  can't be embedded in someone else's iframe for clickjacking),
  `X-Content-Type-Options: nosniff` (a response can't be MIME-sniffed into
  executing as something it isn't), `Referrer-Policy: no-referrer`, and a
  `Content-Security-Policy` that allows scripts only from this origin plus
  the one CDN `scene.html` actually loads Three.js from
  (`cdn.jsdelivr.net`) — not a wildcard.
- **TLS is handled by Render's edge**, not by this app — `https://` on the
  Render-provided domain works automatically; this server only ever speaks
  plain HTTP behind that.
- **Resource exhaustion**: no app-level rate limiting was added — deliberately,
  since the free tier's own 0.1 CPU / 512MB cap already bounds the worst case
  for what is a portfolio demo, not a service with real stakes, and adding
  bespoke rate-limiting code would be complexity without a matching threat.
  Worth revisiting only if this ever serves something with actual value on
  the other end of a request.

**The one thing this review does *not* cover**: this is all about the
*simulated* demo. If real hardware ever gets wired into a publicly-reachable
server the way `main.py` (Phase 3+) eventually will, that's a fundamentally
different threat model — a request that can influence a real motor needs
real authentication, not "the HTTP surface happens to be read-only." Do not
extend this deployment to real hardware without redesigning for that.
