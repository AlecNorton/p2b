#!/usr/bin/env python3
"""Writes a2_contract.json, the contract you fly with.

    python3 tools/make_contract.py

It keeps the camera of scene/policy_contract.json and replaces its action, an
acceleration (for A3 and A5), with a velocity: what your controller outputs.
"""
import argparse
import os
import sys

from splat_hitl.contract import ActionSpec, ControlSpec, PolicyContract

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)


def build(scene_contract_path):
    scene = PolicyContract.load(scene_contract_path)
    return PolicyContract(
        name="a2",
        observation=scene.observation,          # the scene's camera, unchanged
        action=ActionSpec(kind="velocity", frame="body_flu",
                          scale=1.5, yaw_mode="fixed"),
        control=ControlSpec(rate_hz=15.0),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=os.path.join(PACK, "scene"))
    ap.add_argument("--out", default=os.path.join(PACK, "a2_contract.json"))
    a = ap.parse_args(argv)

    src = os.path.join(a.scene, "policy_contract.json")
    if not os.path.exists(src):
        sys.exit("no %s -- is --scene pointing at the scene folder?" % src)

    c = build(src)
    c.save(a.out)
    out = os.path.relpath(a.out)
    if out.startswith(".."):          # outside the cwd: print it in full
        out = a.out
    print("wrote %s" % out)
    print("  action : %s in %s, scale %.2f, yaw %s"
          % (c.action.kind, c.action.frame, c.action.scale, c.action.yaw_mode))
    print("  control: %.1f Hz" % c.control.rate_hz)
    print("  camera : %d x %d, %.2f deg horizontal"
          % (c.observation.sensor.width, c.observation.sensor.height,
             c.observation.sensor.fov_x_deg))
    print("\nfly with:  --contract %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
