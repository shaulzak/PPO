"""Shared helpers for the Biped test scripts (plain scripts: run directly or through run_all.py)."""
import os
import sys

BIPED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BIPED not in sys.path:
    sys.path.insert(0, BIPED)

_failures = []


def check(name, ok, detail=""):
    """Record and print one check. Exits non-zero at finish() if any failed."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        _failures.append(name)


def finish():
    if _failures:
        print(f"  {len(_failures)} failed: {', '.join(_failures)}", flush=True)
        sys.exit(1)
    print("  all passed", flush=True)


_KEEP_ALIVE = []   # the engine frees components whose Python objects die, so scenes stay referenced here


def make_world(objects, gravity=(0, -9.8, 0), tick=1 / 200, epochs=30):
    """A started World whose objects stay referenced for the rest of the test."""
    from bereshit import GameObject, Vector3, World
    gizmos = GameObject()
    world = World(False, list(objects), gizmos, Vector3(*gravity), tick, 1, epochs)
    _KEEP_ALIVE.append((list(objects), gizmos, world))
    world.Start()
    return world
