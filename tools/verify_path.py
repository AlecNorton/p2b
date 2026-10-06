#!/usr/bin/env python3
"""Checks a path against the reconstruction's distance field, not the map.

    python3 tools/verify_path.py path.json [--clearance 0.10]

`path.json` is a list of [x, y, z] in Vicon metres, or a dict with
"waypoints". From code: `check_path(points_m)`.

The path is sampled every 2 cm. It fails if a sample is closer than the
clearance to a surface, or outside the mapped volume: what fails a run.
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np

from splat_hitl.splat_hitl.bundle import SceneBundle

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
STEP_M = 0.02          # sample this finely along each leg


@dataclass
class PathReport:
    ok: bool
    min_clearance_m: float
    worst_point_m: tuple
    n_samples: int
    outside: int
    notes: List[str] = field(default_factory=list)

    def __str__(self):
        head = "PASS" if self.ok else "FAIL"
        s = ("%s  min clearance %.0f mm at (%.2f, %.2f, %.2f), %d samples"
             % (head, self.min_clearance_m * 1000, self.worst_point_m[0],
                self.worst_point_m[1], self.worst_point_m[2], self.n_samples))
        if self.outside:
            s += "\n  %d sample(s) fell outside the mapped volume" % self.outside
        for n in self.notes:
            s += "\n  " + n
        return s


def check_path(points_m, scene=None, clearance_m=0.10, step_m=STEP_M) -> PathReport:
    """Worst clearance along the path, sampled every `step_m`, not just at
    the waypoints."""
    b = SceneBundle.load(scene or os.path.join(PACK, "scene"))
    esdf, tf = b.esdf(), b.transform()
    pts = np.asarray(points_m, dtype=float).reshape(-1, 3)
    if len(pts) < 2:
        raise ValueError("need at least two waypoints, got %d" % len(pts))

    worst, worst_at, n, outside, notes = float("inf"), pts[0], 0, 0, []
    for a, c in zip(pts[:-1], pts[1:]):
        leg = float(np.linalg.norm(c - a))
        for k in range(max(1, int(np.ceil(leg / step_m))) + 1):
            p_v = a + (c - a) * (k / max(1, int(np.ceil(leg / step_m))))
            p_s = tf.point_to_splat(p_v).reshape(3) * tf.metres_per_unit
            r = esdf.at(p_s)
            n += 1
            if getattr(r, "outside", False):
                outside += 1
                continue
            if r.metres < worst:
                worst, worst_at = r.metres, p_v
    if not np.isfinite(worst):
        notes.append("every sample was outside the mapped volume")
        worst = 0.0
    elif worst < clearance_m:
        notes.append("the monitor fails a run below %.0f mm; this path is %.0f mm"
                     % (clearance_m * 1000, worst * 1000))
    return PathReport(ok=(worst >= clearance_m and outside == 0),
                      min_clearance_m=worst, worst_point_m=tuple(np.round(worst_at, 3)),
                      n_samples=n, outside=outside, notes=notes)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="JSON: [[x,y,z], ...] in Vicon metres")
    ap.add_argument("--scene", default=os.path.join(PACK, "scene"))
    ap.add_argument("--clearance", type=float, default=0.10)
    a = ap.parse_args(argv)
    with open(a.path) as fh:
        pts = json.load(fh)
    if isinstance(pts, dict):
        pts = pts.get("waypoints", pts.get("path"))
    rep = check_path(pts, scene=a.scene, clearance_m=a.clearance)
    print(rep)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
