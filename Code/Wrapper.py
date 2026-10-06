#!/usr/bin/env python3
"""P2b entry point: plans a path and writes trajectory.json for policy_template.py.

    python3 Code/Wrapper.py --pair train_1
    python3 Code/Wrapper.py --pair train_1 --pack ../p2b_starter

Fill in the TODO blocks. The starter pack is --pack, else $P2B_PACK, else
../../p2b_starter from this file. Use relative paths: we run this on our machine.
"""
import argparse
import json
import os
import sys

import numpy as np

from splat_hitl.bundle import SceneBundle

from environment import Environment3D
from path_planner import PathPlanner
from trajectory_generator import TrajectoryGenerator

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PACK = os.path.normpath(os.path.join(HERE, "..", "..", "p2b_starter"))


def _is_pack(d):
    return bool(d) and os.path.isdir(os.path.join(d, "scene"))


def resolve_pack(arg):
    for value, source in ((arg, "--pack"), (os.environ.get("P2B_PACK"), "$P2B_PACK")):
        if value is not None:
            if _is_pack(value):
                return os.path.abspath(value)
            sys.exit("%s = %s is not a starter pack: no scene/ inside it."
                     % (source, value))
    if _is_pack(DEFAULT_PACK):
        return DEFAULT_PACK
    sys.exit("no starter pack at %s. Pass --pack or set P2B_PACK." % DEFAULT_PACK)


def load_pair(pack, name):
    with open(os.path.join(pack, "task.json")) as fh:
        task = json.load(fh)
    for p in task["pairs"]:
        if p["name"] == name:
            return np.array(p["start"], float), np.array(p["goal"], float)
    sys.exit("no pair %r in task.json. Have: %s"
             % (name, ", ".join(p["name"] for p in task["pairs"])))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", default=None, help="path to p2b_starter")
    ap.add_argument("--pair", default="train_1", help="pair name in task.json")
    ap.add_argument("--out", default=os.path.join(HERE, "trajectory.json"),
                    help="output file. policy_template.py reads trajectory.json "
                         "from its own folder.")
    ap.add_argument("--clearance", type=float, default=0.10,
                    help="required clearance to the reconstruction, metres")
    ap.add_argument("--inflate", type=float, default=0.10,
                    help="obstacle inflation for planning, metres")
    a = ap.parse_args(argv)

    pack = resolve_pack(a.pack)
    sys.path.insert(0, os.path.join(pack, "tools"))
    from verify_path import check_path                      # noqa: E402

    map_file = os.path.join(pack, "maps", "map1_2b.txt")
    start, goal = load_pair(pack, a.pair)
    print("pack : %s" % pack)
    print("map  : %s" % map_file)
    print("pair : %s  %s -> %s" % (a.pair, start, goal))

    ######## TODO - PARSE THE MAP ########
    # Your P2a environment.py. Vicon metres; the boundary's zmin is 0.30.
    env = Environment3D()
    if(not env.parse_map_file(map_file)):
        sys.exit('Map parsing failed.')
    ######## END TODO ########
    ###DONE###

    ######## TODO - PLAN ########
    # Your P2a planner. Grow the blocks and shrink the boundary by a.inflate:
    # the planner moves a point, and the drone has a 75 mm radius.
    env.safety_margin = a.inflate + .075 #Increasing blocks by a.inflate and .075. Will collide with imaginary inflation based on this.
    planner = PathPlanner(env)

    if(planner.plan_path()):
        waypoints = planner.waypoints
        planner.visualize_tree()
        print(f"Waypoints: {waypoints}")
    else:
        sys.exit("Was not successful path planning.")
    # [[x, y, z], ...] in Vicon metres

    ######## END TODO ########

    ######## TODO - SMOOTH INTO A TRAJECTORY ########
    # The dense path the drone will follow. This is what gets checked.
    traj_gen = TrajectoryGenerator(waypoints)
    try:
        trajectory, time_points, velocities, accelerations = traj_gen.generate_bspline_trajectory()
    except:
        sys.exit("Trajectory generation failed.")
    ######## END TODO ########
    ###DONE###

    if trajectory is None:
        sys.exit("fill in the TODO blocks first.")

    traj = np.asarray(trajectory, float).reshape(-1, 3)
    report = check_path(traj, scene=os.path.join(pack, "scene"),
                        clearance_m=a.clearance)
    print(report)

    # The simulator and ros_node --transform give the policy scene metres.
    tf = SceneBundle.load(os.path.join(pack, "scene")).transform()
    scene = tf.point_to_splat(traj).reshape(-1, 3) * tf.metres_per_unit

    with open(a.out, "w") as fh:
        json.dump({"pair": a.pair,
                   "ok": report.ok,                      # the policy refuses False
                   "min_clearance_m": report.min_clearance_m,
                   "waypoints": traj.round(4).tolist(),             # Vicon metres
                   "scene_waypoints_m": scene.round(4).tolist()},   # scene metres
                  fh, indent=2)
    print("wrote %s  (%d points, %.2f m)"
          % (a.out, len(traj), float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())))
    if not report.ok:
        print("the path fails the check, so the policy will not fly it.")
        return 1
    print("take off to %.2f m and fly with --hold-altitude-m %.2f"
          % (traj[0, 2], traj[0, 2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
