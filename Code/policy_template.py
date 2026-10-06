#!/usr/bin/env python3
"""Flies the trajectory Wrapper.py wrote. Fill in the TODOs.

    ros_node ... --policy policy_template:TrajectoryPolicy

`act(obs, state)` is called at 15 Hz. Return quickly: a tick over 100 ms is
late, and three late ticks in a row hold the drone.

Frames: the simulator and `ros_node --transform` give `state` in scene metres,
so this follows `scene_waypoints_m`. The velocity you return is in the body
frame (x forward, y left, z up), which does not depend on the world frame.

Height: the run starts at --hold-altitude-m, the height your path starts at,
and your vertical velocity moves it from there.
"""
import json
import os

import numpy as np

from splat_hitl.commands import Action
from splat_hitl.policy import Policy

from environment import Environment3D
from path_planner import PathPlanner

HERE = os.path.dirname(os.path.abspath(__file__))


class TrajectoryPolicy(Policy):
    """Follows trajectory.json. Velocity out, nothing else."""

    name = "p2b_trajectory"
    sensor_fingerprint = None          # does not use the camera

    def __init__(self, path=None, speed_ms=0.6, lookahead_m=0.30,
                 arrive_m=0.15, max_start_offset_m=0.5):
        path = path or os.path.join(HERE, "trajectory.json")
        if not os.path.exists(path):
            raise FileNotFoundError("%s does not exist: run Wrapper.py first." % path)
        with open(path) as fh:
            d = json.load(fh)
        if not d.get("ok"):
            raise ValueError("%s did not pass the path check. Fix the plan "
                             "and run Wrapper.py again." % path)
        self.waypoints = np.asarray(d["scene_waypoints_m"], float)   # scene metres
        self.speed_ms = float(speed_ms)
        self.lookahead_m = float(lookahead_m)
        self.arrive_m = float(arrive_m)
        self.max_start_offset_m = float(max_start_offset_m)
        self.i = 0
        self.checked_start = False

    def reset(self):
        """Called once before the run. Reset state here, not in __init__."""
        self.i = 0
        self.checked_start = False

    def act(self, obs, state) -> Action:
        p = np.asarray(state.position_m, float)

        # First tick: the drone must be at the start of the path.
        if not self.checked_start:
            off = float(np.linalg.norm((p - self.waypoints[0])[:2]))
            if off > self.max_start_offset_m:
                raise RuntimeError(
                    "the drone is %.2f m from the start of the path. Put it at "
                    "the start, and run ros_node with --transform." % off)
            self.checked_start = True

        ######## TODO - PICK THE POINT YOU ARE STEERING AT ########
        # Advance self.i past the points already passed, then aim about
        # self.lookahead_m further on. Aiming at the nearest point stalls.
        target = self.waypoints[min(self.i, len(self.waypoints) - 1)]
        ######## END TODO ########

        ######## TODO - THE CONTROL LAW ########
        # World-frame velocity: P on the position error, plus feed-forward
        # along the path. Keep its size within self.speed_ms.
        v_world = np.zeros(3)
        ######## END TODO ########

        # world -> body (x forward, y left); yaw is CCW about +z
        c, s = np.cos(state.yaw_rad), np.sin(state.yaw_rad)
        v_body = np.array([ c * v_world[0] + s * v_world[1],
                           -s * v_world[0] + c * v_world[1],
                            v_world[2]])
        return Action("velocity", "body_flu", v_body, yaw_rate_rad_s=0.0)


class HoverHere(Policy):
    """Zero velocity, no position hold: expect a slow drift. Fly it first each
    session to check the loop."""

    name = "p2b_hover"
    sensor_fingerprint = None

    def act(self, obs, state) -> Action:
        return Action("velocity", "body_flu", np.zeros(3))
