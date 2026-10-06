#!/usr/bin/env python3
"""Checks the A2 setup layer by layer and stops at the first failure.

    python3 tools/hover_check.py

A few seconds, no GPU. It ends with the command for a hover dry run.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)


def step(n, what):
    sys.stdout.write("%d. %-46s" % (n, what))
    sys.stdout.flush()


def ok(msg=""):
    print("ok   %s" % msg)


def die(msg, hint):
    print("FAIL")
    sys.exit("\n   %s\n\n   %s\n" % (msg, hint))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", default=PACK)
    ap.add_argument("--pair", default="train_1")
    ap.add_argument("--clearance", type=float, default=0.10)
    a = ap.parse_args(argv)
    scene_dir = os.path.join(a.pack, "scene")

    step(1, "import numpy")
    try:
        import numpy as np
    except ImportError:
        die("numpy is not installed.", "pip install numpy  (inside your conda env)")
    ok(np.__version__)

    step(2, "import splat_hitl")
    try:
        from splat_hitl.splat_hitl.bundle import SceneBundle
    except ImportError as e:
        die("splat_hitl is not importable: %s" % e,
            "See install.md. The branch is `dev` -- `main` cannot load the\n"
            "   contract this pack ships.")
    ok()

    step(3, "load the scene bundle")
    try:
        b = SceneBundle.load(scene_dir)
    except Exception as e:
        die("SceneBundle.load(%s) raised %s" % (scene_dir, e),
            "Point --pack at the starter pack folder (the one holding scene/).")
    ok()

    step(4, "bundle self-check")
    rep = b.check(clearance_m=a.clearance)
    if not rep.ok:
        die("the scene bundle has errors:\n%s" % rep,
            "Download the starter pack again.")
    ok()
    print(rep)
    print("   (the two warnings are expected: see scene/README.md)")

    step(5, "build the A2 contract")
    sys.path.insert(0, HERE)
    from make_contract import build                        # noqa: E402
    c = build(os.path.join(scene_dir, "policy_contract.json"))
    ok("%s, %.1f Hz, %s in %s"
       % (c.fingerprint()[:12], c.control.rate_hz, c.action.kind, c.action.frame))

    step(6, "clearance at the %s start point" % a.pair)
    with open(os.path.join(a.pack, "task.json")) as fh:
        task = json.load(fh)
    pair = next((p for p in task["pairs"] if p["name"] == a.pair), None)
    if pair is None:
        die("no pair %r in task.json." % a.pair, "Pass --pair with a name from task.json.")
    esdf, tf = b.esdf(), b.transform()
    p_s = tf.point_to_splat(np.array(pair["start"], float)).reshape(3) * tf.metres_per_unit
    r = esdf.at(p_s)
    if getattr(r, "outside", False):
        die("the start point is outside the mapped volume.",
            "Check that scene/vicon_transform.json is the one this pack shipped.")
    ok("%.0f mm" % (r.metres * 1000))

    print("""
Hover dry run: in the lab container, from this folder.

    python3 tools/make_contract.py
    PYTHONPATH=Code:$PYTHONPATH python3 -m splat_hitl.ros_node \\
      --topic     /vicon/<object>/<object> \\
      --contract  a2_contract.json \\
      --transform scene/vicon_transform.json \\
      --esdf      scene/a2_train_esdf.npy \\
      --fake-room 8 4 3 \\
      --policy    policy_template:HoverHere \\
      --hold-altitude-m 0.7 \\
      --log       hover.json \\
      --dry-run

--fake-room replaces the renderer: A2 does not use the camera. --dry-run sends
nothing. To fly, drop --dry-run, with the driver running and the drone
hovering at 0.7 m: see the Deployment page on the course site.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
