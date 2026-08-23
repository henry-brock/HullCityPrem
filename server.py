#!/usr/bin/env python3
"""Hull City PL 2026/27 Predictor — server control script."""

import os
import re
import sys
import signal
import subprocess
import time
import webbrowser
from pathlib import Path

PORT     = 8080
URL      = f"http://localhost:{PORT}"
ROOT     = Path(__file__).parent
PID_FILE = ROOT / ".server.pid"

# ═══════════════════════════════════════════════
# TERMINAL CAPABILITY DETECTION
# 24 = truecolor RGB  |  8 = 256-colour  |  4 = basic 16  |  0 = none
# ═══════════════════════════════════════════════
def _detect() -> int:
    if not sys.stdout.isatty():
        return 0
    ct   = os.environ.get("COLORTERM", "").lower()
    term = os.environ.get("TERM", "")
    prog = os.environ.get("TERM_PROGRAM", "")
    if ct in ("truecolor", "24bit"):
        return 24
    if "256" in term or prog in ("iTerm.app", "vscode", "WezTerm", "kitty", "Hyper"):
        return 8
    if term not in ("dumb", ""):
        return 4
    return 0

DEPTH = _detect()

def _esc(*codes) -> str:
    return f"\033[{';'.join(str(c) for c in codes)}m" if DEPTH else ""

# ── Base modifiers ────────────────────────────
R   = _esc(0)   # reset
B   = _esc(1)   # bold
DIM = _esc(2)   # dim

# ── Hull City: black & amber ──────────────────
#    #F5A12D amber  |  #0A0A0A near-black
if DEPTH >= 24:
    FG_AMBER = _esc("38;2;245;161;45")
    BG_AMBER = _esc("48;2;245;161;45")
    FG_BLACK = _esc("38;2;10;10;10")
    BG_BLACK = _esc("48;2;10;10;10")
elif DEPTH >= 8:
    FG_AMBER = _esc("38;5;214")   # closest 256-colour amber
    BG_AMBER = _esc("48;5;214")
    FG_BLACK = _esc("38;5;232")   # very dark grey
    BG_BLACK = _esc("48;5;232")
else:
    FG_AMBER = _esc(33)           # yellow — nearest basic-16 to amber
    BG_AMBER = _esc(43)
    FG_BLACK = _esc(30)
    BG_BLACK = _esc(40)

# ── Supporting colours ────────────────────────
FG_WHITE = _esc(97)
FG_GREEN = _esc(92)
FG_RED   = _esc(91)
FG_GREY  = _esc(90)

# ═══════════════════════════════════════════════
# BOX-DRAWING HELPERS
# ═══════════════════════════════════════════════
W = 47   # inner menu width

def _strip(s: str) -> str:
    return re.sub(r"\033\[[^m]*m", "", s)

def _line(ch="─", n=W + 2) -> str:
    return ch * n

def _box_top() -> str: return f"{FG_AMBER}╔{_line('═')}╗{R}"
def _box_bot() -> str: return f"{FG_AMBER}╚{_line('═')}╝{R}"
def _box_mid() -> str: return f"{FG_AMBER}╠{_line('═')}╣{R}"

def _row(text: str = "", pad: int = W) -> str:
    visible = len(_strip(text))
    spaces  = max(0, pad - visible)
    return f"{FG_AMBER}║{R} {text}{' ' * spaces} {FG_AMBER}║{R}"

# ═══════════════════════════════════════════════
# PROCESS HELPERS
# ═══════════════════════════════════════════════
def is_running() -> bool:
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        PID_FILE.unlink(missing_ok=True)
        return False

def _pid() -> str:
    return PID_FILE.read_text().strip() if PID_FILE.exists() else "—"

# ═══════════════════════════════════════════════
# CORE ACTIONS
# ═══════════════════════════════════════════════
def start(quiet: bool = False) -> bool:
    if is_running():
        if not quiet:
            print(f"  {FG_AMBER}⚠  {FG_WHITE}Already running{R}  "
                  f"{FG_WHITE}{DIM}PID {_pid()}  │  {URL}{R}")
        return True

    proc = subprocess.Popen(
        [sys.executable, ROOT / "app.py"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    PID_FILE.write_text(str(proc.pid))
    time.sleep(0.6)

    if is_running():
        if not quiet:
            print(f"  {FG_GREEN}●  {FG_WHITE}Server started{R}  "
                  f"{FG_WHITE}{DIM}PID {_pid()}  │  {FG_AMBER}{URL}{R}")
        return True

    print(f"  {FG_RED}✖  {FG_WHITE}Failed to start — is port {PORT} in use?{R}")
    PID_FILE.unlink(missing_ok=True)
    return False


def stop(quiet: bool = False) -> None:
    if not is_running():
        if not quiet:
            print(f"  {FG_AMBER}⚠  Server is not running{R}")
        return

    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.3)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass

    PID_FILE.unlink(missing_ok=True)
    if not quiet:
        print(f"  {FG_RED}■  {FG_WHITE}Server stopped{R}")


def status_line() -> str:
    if is_running():
        badge = f"{BG_AMBER}{FG_BLACK}{B} ● RUNNING {R}"
        return f"{badge}  {FG_WHITE}{DIM}PID {_pid()}  │  {URL}{R}"
    return f"{FG_RED}{B}■ STOPPED{R}"

# ═══════════════════════════════════════════════
# MENU
# ═══════════════════════════════════════════════
def _print_menu() -> None:
    bi = W + 2   # banner inner width

    # ── Banner ──
    print()
    print(f"{BG_BLACK}{FG_AMBER}{B}{'':^{bi}}{R}")
    print(f"{BG_AMBER}{FG_BLACK}{B}{'  HULL CITY — PREMIER LEAGUE  ':^{bi}}{R}")
    print(f"{BG_BLACK}{FG_WHITE}{'  2026/27 Season Predictor  ':^{bi}}{R}")
    print(f"{BG_BLACK}{FG_AMBER}{B}{'':^{bi}}{R}")

    # ── Status ──
    print(_box_top())
    print(_row())
    print(_row(f"  {FG_WHITE}Status{R}   {status_line()}"))
    print(_row())
    print(_box_mid())

    # ── Options ──
    opts = [
        ("1", "Start server",    FG_GREEN),
        ("2", "Stop server",     FG_RED),
        ("3", "Restart server",  FG_AMBER),
        ("4", "Open in browser", FG_WHITE),
        ("q", "Quit",            FG_WHITE),
    ]

    print(_row())
    for key, label, colour in opts:
        key_tag = f"{BG_AMBER}{FG_BLACK}{B} {key} {R}"
        print(_row(f"   {key_tag}  {colour}{label}{R}"))
    print(_row())
    print(_box_bot())
    print()


def menu() -> None:
    _print_menu()
    try:
        choice = input(f"  {FG_AMBER}{B}›{R} ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print(f"\n  {FG_GREY}Bye.{R}\n")
        sys.exit(0)

    print()
    running = is_running()

    if choice == "1":
        start()
    elif choice == "2":
        stop()
    elif choice == "3":
        stop(quiet=True)
        time.sleep(0.3)
        if start(quiet=True):
            print(f"  {FG_AMBER}↺  {FG_WHITE}Restarted{R}  "
                  f"{FG_WHITE}{DIM}PID {_pid()}  │  {FG_AMBER}{URL}{R}")
    elif choice == "4":
        if running:
            webbrowser.open(URL)
            print(f"  {FG_AMBER}↗  {FG_WHITE}Opening{R}  {FG_AMBER}{URL}{R}")
        else:
            print(f"  {FG_AMBER}⚠  {FG_WHITE}Start the server first{R}")
    elif choice in ("q", "quit", "exit", ""):
        print(f"  {FG_GREY}Bye.{R}")
        sys.exit(0)
    else:
        print(f"  {FG_AMBER}⚠  Unknown option '{choice}'{R}")

    print()

# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════
COMMANDS = {
    "start":   lambda: start(),
    "stop":    lambda: stop(),
    "restart": lambda: (stop(quiet=True), time.sleep(0.3), start()),
    "status":  lambda: print(f"  {status_line()}"),
}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg is None:
        menu()
    elif arg in COMMANDS:
        print()
        COMMANDS[arg]()
        print()
    else:
        keys = "|".join(COMMANDS)
        print(f"\n  Usage: python3 server.py [{keys}]\n")
        sys.exit(1)
