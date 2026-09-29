"""
admin.py — operations panel for the predictor, on its own IP.

Why a second server
───────────────────
Every update this app needs is a long, noisy, occasionally destructive job:
re-fetching a season of lineups takes minutes, rebuilding Newcache hits the
Pulse API 380 times, and picking up a code change means killing the very
process that would have to serve the button you clicked. None of that belongs
inside app.py — a server cannot restart itself, and a 4-minute refetch would
block the page it was launched from.

So this runs as a separate process on 127.0.0.2:8080. Same port, different
IP: the main app keeps 127.0.0.1:8080 untouched, and 127.0.0.0/8 is loopback,
so this is reachable from this machine and nowhere else. Binding it to a
routable address would hand process control to the network, so --host refuses
anything outside loopback unless --unsafe-host is passed too.

Safety
──────
Actions are a fixed table, looked up by key. Nothing here takes a command, a
path or a shell string from the request — the worst a caller can do is run one
of the jobs listed below, which is the entire point of the panel.
"""

import argparse
import ipaddress
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, request

ROOT       = Path(__file__).parent
APP_SCRIPT = (ROOT / "app.py").resolve()
CACHE      = ROOT / "cache"
APP_LOG    = ROOT / "app.log"
SEASON     = "2627"
MAIN_HOST  = "127.0.0.1"
MAIN_PORT  = 8080

app = Flask(__name__)


# ── the main app process ────────────────────────────────────────────────
def _app_pids() -> list[int]:
    """PIDs running app.py, matched on the resolved script path.

    Read from /proc rather than shelling out to pgrep: a name match would also
    catch an editor holding the file open, and the path has to be resolved
    against each process's own cwd because `python3 app.py` and the absolute
    form are the same server started two different ways.
    """
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = entry.joinpath("cmdline").read_bytes().decode().split("\0")
        except OSError:
            continue                      # process exited mid-scan
        argv = [a for a in argv if a]
        if len(argv) < 2 or not argv[-1].endswith("app.py"):
            continue
        script = Path(argv[-1])
        if not script.is_absolute():
            try:
                script = Path(os.readlink(entry / "cwd")) / script
            except OSError:
                continue
        try:
            if script.resolve() == APP_SCRIPT:
                pids.append(int(entry.name))
        except OSError:
            continue
    return pids


def _proc_started(pid: int):
    try:
        return (Path("/proc") / str(pid)).stat().st_mtime
    except OSError:
        return None


def _port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    with socket.socket() as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def _stop_app(sig=signal.SIGTERM, wait=8.0) -> list[int]:
    pids = _app_pids()
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
    deadline = time.time() + wait
    while time.time() < deadline and _app_pids():
        time.sleep(0.2)
    stubborn = _app_pids()
    for pid in stubborn:
        print(f"pid {pid} ignored SIGTERM, sending SIGKILL")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return pids


def _start_app() -> int:
    # start_new_session detaches it from this panel: restarting or stopping
    # admin.py must never take the main app down with it.
    log = APP_LOG.open("ab")
    proc = subprocess.Popen(
        [sys.executable, str(APP_SCRIPT)],
        cwd=str(ROOT), stdout=log, stderr=log, start_new_session=True,
    )
    for _ in range(50):
        if _port_open(MAIN_HOST, MAIN_PORT):
            break
        time.sleep(0.2)
    return proc.pid


# ── job runner ──────────────────────────────────────────────────────────
# One job at a time. Two refetches of the same season racing each other would
# interleave writes to the same cache file, and the log would be unreadable.
_lock = threading.Lock()
_job = {"name": None, "running": False, "started": None,
        "finished": None, "ok": None, "log": []}


class _LogStream:
    """Line-buffers whatever a job prints into the job log.

    The fetchers report progress on stderr ("[lineups] 2627: 150/380"), which
    is the only visibility into a job that runs for minutes — so it is
    captured rather than left in the terminal nobody is watching.
    """

    def __init__(self):
        self._buf = ""

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                _append(line.rstrip())

    def flush(self):
        if self._buf.strip():
            _append(self._buf.rstrip())
        self._buf = ""


def _append(line: str) -> None:
    _job["log"].append(f"{datetime.now():%H:%M:%S}  {line}")
    del _job["log"][:-400]


def _run(name: str, fn) -> bool:
    if not _lock.acquire(blocking=False):
        return False
    _job.update(name=name, running=True, started=time.time(),
                finished=None, ok=None, log=[])

    def worker():
        stream = _LogStream()
        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                fn()
                stream.flush()
            _job["ok"] = True
            _append(f"✓ {name} finished")
        except Exception as exc:
            stream.flush()
            _job["ok"] = False
            _append(f"✗ {name} failed: {type(exc).__name__}: {exc}")
        finally:
            _job["running"] = False
            _job["finished"] = time.time()
            _lock.release()

    threading.Thread(target=worker, daemon=True).start()
    return True


# ── actions ─────────────────────────────────────────────────────────────
def _do_restart():
    stopped = _stop_app()
    print(f"stopped {stopped}" if stopped else "nothing was running")
    pid = _start_app()
    up = _port_open(MAIN_HOST, MAIN_PORT)
    print(f"started pid {pid} — {MAIN_HOST}:{MAIN_PORT} {'up' if up else 'NOT RESPONDING'}")


def _do_stop():
    stopped = _stop_app()
    print(f"stopped {stopped}" if stopped else "nothing was running")


def _do_start():
    if _app_pids():
        print("already running — use Restart")
        return
    print(f"started pid {_start_app()}")


def _do_fixtures():
    import fixture_scraper
    print(f"{len(fixture_scraper.fetch(force=True))} fixtures")


def _do_results():
    import pulse_results
    print(f"{len(pulse_results.fetch(SEASON, force=True))} matches")


def _do_lineups():
    # Not force: a finished match's lineup and events never change, so only
    # matches missing from the cache are fetched. force re-downloads the whole
    # season and is only needed after the cache format changes.
    import lineups
    print(f"{len(lineups.fetch(SEASON))} matches")


def _do_players():
    import pl_players
    import pulse_players
    print(f"{len(pl_players.fetch(season=SEASON, force=True)['players'])} league players")
    print(f"{len(pulse_players.fetch(season=SEASON, force=True).get('players', []))} Hull players")


def _do_newcache():
    # Incremental: only new, unsettled-and-stale, or newly settled matches cost
    # a request (see pulse_results.match_stats).
    import newcache_builder
    data = newcache_builder.build_season(SEASON)
    print(f"{data['matches']} matches in Newcache")


def _do_import():
    """Merge scraped results into data.json, via the main app.

    Deliberately an HTTP call rather than a copy of the merge: /api/import
    also backfills missing fields on stored matches and drops the player
    caches when something new lands, and two implementations of that would
    drift.
    """
    body = json.dumps({"season": SEASON, "force": True}).encode()
    req = urllib.request.Request(
        f"http://{MAIN_HOST}:{MAIN_PORT}/api/import", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"main app unreachable ({exc.reason}) — start it first") from None
    print(f"added {d.get('added')} · enriched {d.get('enriched')} · total {d.get('total')}")
    for m in d.get("matches", []):
        print(f"  + {m.get('homeId')} {m.get('homeGoals')}–{m.get('awayGoals')} {m.get('awayId')}")


def _do_everything():
    for label, fn in (("fixtures", _do_fixtures), ("results", _do_results),
                      ("newcache", _do_newcache), ("lineups", _do_lineups),
                      ("players", _do_players), ("import", _do_import)):
        print(f"── {label}")
        fn()


ACTIONS = {
    "restart":  ("Restart app",      _do_restart,  "Pick up code changes — stop and relaunch app.py"),
    "start":    ("Start app",        _do_start,    "Launch app.py if it is not running"),
    "stop":     ("Stop app",         _do_stop,     "Shut the main app down"),
    "import":   ("Sync results",     _do_import,   "Scrape results and merge them into data.json"),
    "fixtures": ("Refresh fixtures", _do_fixtures, "Re-fetch the official fixture list"),
    "results":  ("Refresh results",  _do_results,  "Re-fetch this season's Pulse results"),
    "newcache": ("Build Newcache",   _do_newcache, "Fetch Opta stat blocks for any new matches"),
    "lineups":  ("Refresh lineups",  _do_lineups,  "Fetch formations, events and lineups for new matches"),
    "players":  ("Refresh players",  _do_players,  "Re-fetch league and Hull player season stats"),
    "all":      ("Update everything", _do_everything, "Every refresh above, in order (slow)"),
}

DESTRUCTIVE = {"stop", "restart", "all", "lineups"}


# ── api ─────────────────────────────────────────────────────────────────
@app.post("/api/run/<key>")
def run_action(key):
    entry = ACTIONS.get(key)
    if not entry:
        return jsonify({"error": "unknown action"}), 404
    label, fn, _ = entry
    if not _run(label, fn):
        return jsonify({"error": f"'{_job['name']}' is still running"}), 409
    return jsonify({"ok": True, "started": label})


@app.get("/api/job")
def job():
    return jsonify(_job)


# Deductions are quick file edits, not jobs, so they skip the job lock — a
# lineup refetch that takes four minutes should not block docking a club.
@app.get("/api/deductions")
def deductions_list():
    import deductions
    entries = deductions.load()
    return jsonify({"deductions": entries, "totals": deductions.totals(entries),
                    "teams": deductions.team_names(), "max": deductions.MAX_POINTS})


@app.post("/api/deductions")
def deductions_add():
    import deductions
    body = request.get_json(silent=True) or {}
    try:
        entry = deductions.add(body.get("team"), body.get("points"), body.get("reason") or "")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True, "deduction": entry})


@app.patch("/api/deductions/<entry_id>")
def deductions_update(entry_id):
    import deductions
    body = request.get_json(silent=True) or {}
    try:
        entry = deductions.update(entry_id, body.get("points"), body.get("reason"))
    except KeyError:
        return jsonify({"error": "no such deduction"}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True, "deduction": entry})


@app.delete("/api/deductions/<entry_id>")
def deductions_remove(entry_id):
    import deductions
    try:
        deductions.remove(entry_id)
    except KeyError:
        return jsonify({"error": "no such deduction"}), 404
    return jsonify({"ok": True})


@app.get("/api/status")
def status():
    pids = _app_pids()
    started = _proc_started(pids[0]) if pids else None
    caches = []
    for f in sorted(CACHE.glob("*.json")) + sorted((ROOT / "Newcache").glob("*.json")):
        try:
            st = f.stat()
        except OSError:
            continue
        caches.append({"name": f.name, "kb": round(st.st_size / 1024),
                       "age": round(time.time() - st.st_mtime)})
    caches.sort(key=lambda c: c["age"])
    return jsonify({
        "running": bool(pids), "pids": pids,
        "uptime": round(time.time() - started) if started else None,
        "port": _port_open(MAIN_HOST, MAIN_PORT),
        "main": f"http://{MAIN_HOST}:{MAIN_PORT}/",
        "actions": [{"key": k, "label": v[0], "note": v[2], "warn": k in DESTRUCTIVE}
                    for k, v in ACTIONS.items()],
        "caches": caches[:24],
    })


@app.get("/")
def panel():
    return Response(PANEL, mimetype="text/html")


PANEL = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Predictor admin</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root { --bg:#0e1116; --card:#161a21; --line:rgba(255,255,255,.09);
          --text:#e6e8eb; --dim:#8b939e; --amber:#F5A12D; --green:#00ff85; --red:#ff4d5e; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
  header { padding:14px 20px; border-bottom:1px solid var(--line); display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:.95rem; margin:0; letter-spacing:.08em; text-transform:uppercase; }
  .dot { width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:6px; }
  .up { background:var(--green); box-shadow:0 0 8px var(--green); } .down { background:var(--red); }
  .muted { color:var(--dim); }
  main { padding:20px; display:grid; gap:18px; grid-template-columns:minmax(300px,1fr) minmax(360px,1.4fr); }
  @media (max-width:860px) { main { grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:9px; padding:14px; }
  .card h2 { font-size:.68rem; letter-spacing:.09em; text-transform:uppercase; color:var(--dim); margin:0 0 11px; }
  button { width:100%; text-align:left; background:rgba(255,255,255,.04); color:var(--text);
           border:1px solid var(--line); border-radius:7px; padding:9px 11px; margin-bottom:7px;
           font:inherit; cursor:pointer; transition:background .12s,border-color .12s; }
  button:hover:not(:disabled) { background:rgba(245,161,45,.12); border-color:rgba(245,161,45,.45); }
  button:disabled { opacity:.45; cursor:not-allowed; }
  button.warn:hover:not(:disabled) { background:rgba(255,77,94,.12); border-color:rgba(255,77,94,.45); }
  button b { display:block; font-weight:700; }
  button span { color:var(--dim); font-size:.74rem; }
  pre { background:#0a0c10; border:1px solid var(--line); border-radius:7px; padding:11px;
        margin:0; height:calc(100vh - 190px); min-height:320px; overflow:auto;
        font-size:.76rem; line-height:1.55; white-space:pre-wrap; }
  table { width:100%; border-collapse:collapse; font-size:.75rem; }
  td { padding:2px 0; } td:last-child { text-align:right; color:var(--dim); }
  .run { color:var(--amber); } .ok { color:var(--green); } .bad { color:var(--red); }
  #ded-form { display:grid; grid-template-columns:1.3fr .55fr 1.6fr auto; gap:6px; }
  #ded-form select, #ded-form input { background:#0a0c10; color:var(--text); border:1px solid var(--line);
      border-radius:6px; padding:7px 8px; font:inherit; font-size:.8rem; min-width:0; }
  button.small { width:auto; margin:0; padding:7px 12px; text-align:center; }
  .ded { display:flex; gap:10px; align-items:center; padding:7px 0; border-top:1px solid var(--line); font-size:.8rem; }
  .ded .pts { color:var(--red); font-weight:800; min-width:44px; }
  .ded .why { flex:1; color:var(--dim); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .ded button { width:auto; margin:0; padding:3px 8px; font-size:.7rem; }
</style></head><body>
<header>
  <h1>Predictor admin</h1>
  <span id="state" class="muted">…</span>
  <a id="link" class="muted" href="#" target="_blank" style="margin-left:auto;color:var(--amber)">open app ↗</a>
</header>
<main>
  <div>
    <div class="card"><h2>Actions</h2><div id="actions"></div></div>
    <div class="card" style="margin-top:18px">
      <h2>Points deductions</h2>
      <form id="ded-form" onsubmit="addDeduction(event)">
        <select id="ded-team" required></select>
        <input id="ded-points" type="number" min="1" step="1" placeholder="pts" required>
        <input id="ded-reason" type="text" maxlength="200" placeholder="reason (optional)">
        <button type="submit" class="small">Dock</button>
      </form>
      <div id="ded-msg" class="muted" style="font-size:.74rem;min-height:1.2em;margin:6px 0"></div>
      <div id="ded-list"></div>
      <div class="muted" style="font-size:.72rem;margin-top:8px">The app picks changes up when its tab is
        next focused. Re-run the simulation afterwards — existing odds were computed without it.</div>
    </div>
    <div class="card" style="margin-top:18px"><h2>Freshest caches</h2><table id="caches"></table></div>
  </div>
  <div class="card"><h2>Job — <span id="jobname" class="muted">idle</span></h2><pre id="log">No job has run yet.</pre></div>
</main>
<script>
let busy = false;

function age(s) {
  if (s == null) return '–';
  if (s < 90) return s + 's';
  if (s < 5400) return Math.round(s / 60) + 'm';
  if (s < 172800) return Math.round(s / 3600) + 'h';
  return Math.round(s / 86400) + 'd';
}

async function poll() {
  const [st, job] = await Promise.all([
    fetch('/api/status').then(r => r.json()),
    fetch('/api/job').then(r => r.json()),
  ]);

  document.getElementById('state').innerHTML = st.running
    ? `<span class="dot up"></span>app up · pid ${st.pids.join(', ')} · ${age(st.uptime)}`
       + (st.port ? '' : ' <span class="bad">· port not answering</span>')
    : `<span class="dot down"></span>app not running`;
  document.getElementById('link').href = st.main;

  const box = document.getElementById('actions');
  if (box.dataset.built !== '1') {
    box.innerHTML = st.actions.map(a =>
      `<button class="${a.warn ? 'warn' : ''}" data-key="${a.key}" onclick="run('${a.key}')">
         <b>${a.label}</b><span>${a.note}</span></button>`).join('');
    box.dataset.built = '1';
  }

  busy = job.running;
  box.querySelectorAll('button').forEach(b => { b.disabled = busy; });

  const name = document.getElementById('jobname');
  if (job.name) {
    const cls = job.running ? 'run' : job.ok ? 'ok' : 'bad';
    const tail = job.running ? 'running…' : job.ok ? 'done' : 'failed';
    name.className = cls;
    name.textContent = `${job.name} — ${tail}`;
  }
  const log = document.getElementById('log');
  if (job.log && job.log.length) {
    const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
    log.textContent = job.log.join('\\n');
    if (atEnd) log.scrollTop = log.scrollHeight;
  }

  document.getElementById('caches').innerHTML = st.caches.map(c =>
    `<tr><td>${c.name}</td><td>${c.kb} kB · ${age(c.age)} ago</td></tr>`).join('');
}

async function run(key) {
  if (busy) return;
  const btn = document.querySelector(`[data-key="${key}"]`);
  if (btn.classList.contains('warn') &&
      !confirm(`${btn.querySelector('b').textContent} — go ahead?`)) return;
  busy = true;
  await fetch('/api/run/' + key, { method: 'POST' });
  poll();
}

// ── deductions ──
// Reasons are typed by hand and rendered as HTML, so everything goes through esc().
const esc = s => String(s ?? '').replace(/[&<>"']/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
let dedTeams = {};

function dedMsg(text, bad) {
  const el = document.getElementById('ded-msg');
  el.textContent = text;
  el.className = bad ? 'bad' : 'muted';
}

async function loadDeductions() {
  const d = await fetch('/api/deductions').then(r => r.json());
  dedTeams = d.teams;
  const sel = document.getElementById('ded-team');
  if (!sel.options.length) {
    sel.innerHTML = '<option value="" disabled selected>team…</option>' +
      Object.entries(d.teams).map(([id, n]) => `<option value="${esc(id)}">${esc(n)}</option>`).join('');
    document.getElementById('ded-points').max = d.max;
  }
  document.getElementById('ded-list').innerHTML = d.deductions.length
    ? d.deductions.map(x => `<div class="ded">
        <span class="pts">−${x.points}</span>
        <span>${esc(d.teams[x.team] || x.team)}</span>
        <span class="why" title="${esc(x.reason)}">${esc(x.reason) || '—'} · ${esc(x.added)}</span>
        <button onclick="editDeduction('${esc(x.id)}', ${Number(x.points)})">edit</button>
        <button class="warn" onclick="removeDeduction('${esc(x.id)}')">remove</button>
      </div>`).join('')
    : '<div class="muted" style="font-size:.78rem">None — every club is on the points it earned.</div>';
}

async function sendDeduction(url, method, body) {
  const r = await fetch(url, { method, headers: { 'Content-Type': 'application/json' },
                               body: body ? JSON.stringify(body) : undefined });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) { dedMsg(d.error || `failed (${r.status})`, true); return false; }
  return true;
}

async function addDeduction(ev) {
  ev.preventDefault();
  const team   = document.getElementById('ded-team').value;
  const points = Number(document.getElementById('ded-points').value);
  const reason = document.getElementById('ded-reason').value;
  if (!confirm(`Dock ${dedTeams[team] || team} ${points} point${points === 1 ? '' : 's'}?`)) return;
  if (!await sendDeduction('/api/deductions', 'POST', { team, points, reason })) return;
  dedMsg(`Docked ${dedTeams[team]} ${points} pts.`);
  ev.target.reset();
  loadDeductions();
}

async function editDeduction(id, current) {
  const v = prompt('Points for this deduction (e.g. reduced on appeal):', current);
  if (v === null) return;
  if (!await sendDeduction('/api/deductions/' + id, 'PATCH', { points: Number(v) })) return;
  dedMsg('Updated.');
  loadDeductions();
}

async function removeDeduction(id) {
  if (!confirm('Remove this deduction? The points go back.')) return;
  if (!await sendDeduction('/api/deductions/' + id, 'DELETE')) return;
  dedMsg('Removed.');
  loadDeductions();
}

loadDeductions();
poll();
setInterval(poll, 1500);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description="Admin panel for the predictor")
    ap.add_argument("--host", default="127.0.0.2")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--unsafe-host", action="store_true",
                    help="allow binding a non-loopback address (do not)")
    args = ap.parse_args()

    if not ipaddress.ip_address(args.host).is_loopback and not args.unsafe_host:
        raise SystemExit(
            f"refusing to bind {args.host}: this panel restarts processes, so it stays "
            f"on loopback. Pass --unsafe-host if you really mean it.")

    print(f"admin panel  →  http://{args.host}:{args.port}/")
    print(f"main app     →  http://{MAIN_HOST}:{MAIN_PORT}/")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
