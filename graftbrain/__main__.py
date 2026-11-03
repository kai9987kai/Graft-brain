"""GraftBrain CLI.

    python -m graftbrain run                     # build/resume, open the live 3D viewer
    python -m graftbrain run --webcam            # ... with the live webcam feed
    python -m graftbrain train --steps 20000     # headless, as fast as possible, autosaving
    python -m graftbrain info                    # describe the grafted topology
    python -m graftbrain fetch-flywire           # download FlyWire v783 connections (~1 GB, CC-BY-NC)
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time

DEFAULT_SPECS = ["proxy:flywire", "proxy:hemibrain", "proxy:malecns", "proxy:banc",
                 "proxy:optic-lobe", "proxy:manc", "proxy:fanc", "proxy:larval"]


def _common(p):
    p.add_argument("--connectome", "-c", action="append", dest="specs",
                   help="connectome spec (repeatable). proxy:<name> | flywire:<file> | neuprint:<dataset> | csv:<file>")
    p.add_argument("--max-neurons", type=int, default=1500, help="per-connectome neuron cap (real-time budget)")
    p.add_argument("--state", default="state/graftbrain.pt", help="checkpoint path (auto-resumed)")
    p.add_argument("--fresh", action="store_true", help="ignore any saved state and build a new brain")
    p.add_argument("--task", default="auto", help="auto (curriculum) or a fixed task name")
    p.add_argument("--webcam", action="store_true", help="use the live webcam (frames stay local, never saved)")
    p.add_argument("--autosave", type=float, default=60, help="seconds between autosaves")
    p.add_argument("--microsteps", type=int, default=4, help="neural microsteps per world step")
    p.add_argument("--seed", type=int, default=0)


def _sim(a):
    from .runtime import Simulation
    return Simulation(a.specs or DEFAULT_SPECS, a.state, a.max_neurons, a.webcam, a.autosave, a.task, a.seed,
                      a.microsteps, a.fresh)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="graftbrain", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="live closed loop + 3D viewer"); _common(r); r.add_argument("--port", type=int, default=8770)
    t = sub.add_parser("train", help="headless training"); _common(t); t.add_argument("--steps", type=int, default=10000)
    i = sub.add_parser("info", help="describe topology"); _common(i)
    f = sub.add_parser("fetch-flywire", help="download FlyWire v783 connections")
    f.add_argument("--out", default="data/proofread_connections_783.feather")
    a = ap.parse_args(argv)

    if a.cmd == "fetch-flywire":
        import urllib.request
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        url = "https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1"
        print(f"downloading {url}\n -> {a.out}\nFlyWire data is CC-BY-NC-4.0: cite Dorkenwald et al. 2024 & Schlegel et al. 2024.")
        urllib.request.urlretrieve(url, a.out, lambda b, bs, tot: print(f"\r{b * bs / 1e6:8.1f} / {tot / 1e6:.1f} MB", end=""))
        print("\ndone. use:  python -m graftbrain run -c flywire:" + a.out + " -c proxy:manc ...")
        return

    if a.cmd == "info":
        from . import connectomes as cx
        from .graft import build
        g = build([cx.load(s, a.max_neurons, a.seed) for s in (a.specs or DEFAULT_SPECS)], seed=a.seed)
        d = g.describe()
        print(json.dumps({k: v for k, v in d.items() if k != "sites"}, indent=2, default=str))
        kinds = {}
        for s in d["sites"]:
            kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
        print(f"\n{len(d['sites'])} graft sites by kind: {kinds}")
        for s in d["sites"][:25]:
            print(f"  #{s['id']:3d} {s['kind']:9s} {s['from']:>10s} -> {s['to']:<10s} {s['edges']:5d} edges  {s['detail']}")
        if len(d["sites"]) > 25:
            print(f"  ... {len(d['sites']) - 25} more")
        return

    if a.cmd == "train":
        sim = _sim(a)
        t0 = time.time(); last = t0
        try:
            for k in range(a.steps):
                sim.step_once()
                if time.time() - sim.last_save > a.autosave:
                    sim.save()
                if time.time() - last > 10:
                    last = time.time()
                    sr = {t: round(v, 2) for t, v in sim.world.success_rates().items() if v is not None}
                    print(f"step {sim.total_steps:7d} | {k / (last - t0):5.1f} steps/s | ep {sim.world.episode} | "
                          f"loss {sim.agent.stats['loss']:.3f} | success {sr}")
        except KeyboardInterrupt:
            print("\ninterrupted")
        sim.save("end of training"); sim.cam.close()
        return

    if a.cmd in (None, "run"):
        if a.cmd is None:
            a = ap.parse_args(["run"] + (argv or sys.argv[1:]))
        from .server import serve
        sim = _sim(a)
        sim.start()
        try:
            serve(sim, a.port)
        finally:
            sim.stop()


if __name__ == "__main__":
    main()
