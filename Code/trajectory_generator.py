import numpy as np
from scipy.interpolate import splprep, splev, BSpline
import matplotlib.pyplot as plt

class TrajectoryGenerator:
    """
    Generate smooth trajectory from waypoints using
splines
    Complete implementation with velocity and acceleration profiles
    """

    def __init__(self, waypoints):
        self.waypoints = np.array(waypoints)
        self.trajectory_duration = None  # seconds
        self.max_velocity = None  # m/s
        self.max_acceleration = None  # m/s^2
        self.vmax=40
        self.amax=5
        self.f=0.45

    ##############################################################
    #### TODO - Implement spline trajectory generation ###########
    #### TODO - Ensure velocity and acceleration constraints #####
    #### TODO - Add member functions as needed ###################
    ##############################################################

    def generate_bspline_trajectory(self, num_points, vmax=2, amax=4,
                                    env=None, verbose=True):
        
        print("Generating spline trajectory...")
        trajectory_points = None
        time_points = None
        velocities = None
        accelerations = None

        ############## IMPLEMENTATION STARTS HERE ##############

        wp = np.array(self.waypoints, dtype=float)
        
        traj = generate_trajectory(wp,vmax,amax,self.f,env,verbose=verbose)
        self.trajectory_duration = traj.duration
        self.max_velocity = vmax
        self.max_acceleration = amax

        time_points, trajectory_points, velocities, accelerations = traj.sample_grid(int(num_points))
        return trajectory_points, time_points, velocities, accelerations

    def visualize_trajectory(self, trajectory_points=None, velocities=None,
                           accelerations=None, ax=None):
        """Visualize the trajectory with velocity and acceleration vectors"""
        if ax is None:
            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            standalone = True
        else:
            ax1 = ax
            standalone = False

        if trajectory_points is not None:
            # Plot 3D trajectory
            ax1.plot(trajectory_points[:, 0], trajectory_points[:, 1],
                    trajectory_points[:, 2], 'b-', linewidth=2, label='Spline Trajectory')

            # Plot waypoints
            ax1.plot(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2],
                    'ro-', markersize=8, linewidth=2, label='Waypoints')

            # Plot velocity vectors (sampled)
            if velocities is not None:
                step = max(1, len(trajectory_points) // 20)  # Show ~20 vectors
                for i in range(0, len(trajectory_points), step):
                    pos = trajectory_points[i]
                    vel = velocities[i] * 0.5  # Scale for visualization
                    ax1.quiver(pos[0], pos[1], pos[2],
                             vel[0], vel[1], vel[2],
                             color='green', alpha=0.7, arrow_length_ratio=0.1)

            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title('3D Trajectory')
            ax1.legend()

        if standalone and velocities is not None and accelerations is not None:
            # Plot velocity magnitude over time
            time_points = np.linspace(0, self.trajectory_duration, len(velocities))
            vel_magnitudes = np.linalg.norm(velocities, axis=1)
            ax2.plot(time_points, vel_magnitudes, 'g-', linewidth=2)
            ax2.axhline(y=self.max_velocity, color='r', linestyle='--',
                       label=f'Max Vel: {self.max_velocity} m/s')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Velocity (m/s)')
            ax2.set_title('Velocity Profile')
            ax2.grid(True)
            ax2.legend()

            # Plot acceleration magnitude over time
            acc_magnitudes = np.linalg.norm(accelerations, axis=1)
            ax3.plot(time_points, acc_magnitudes, 'm-', linewidth=2)
            ax3.axhline(y=self.max_acceleration, color='r', linestyle='--',
                       label=f'Max Acc: {self.max_acceleration} m/s²')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Acceleration (m/s²)')
            ax3.set_title('Acceleration Profile')
            ax3.grid(True)
            ax3.legend()

            plt.tight_layout()
            plt.show()

        return ax1 if not standalone else None




V_AVG            = 0.5    # heuristic average speed for time allocation [m/s]
COLLISION_STEP   = 0.1    # arc-length sample step [m], < env.safety_margin (0.5)
MAX_REPAIR_ITERS = 5     # cap on reactive-insertion iterations
_REG             = 1e-9   # tiny Tikhonov regularization for the QP solve


def _bc_matrix(T):
    """M such that M @ coeffs = bc. coeffs = inv(M) @ bc."""
    T2, T3, T4, T5 = T*T, T**3, T**4, T**5
    return np.array([
        [1, 0, 0,    0,     0,      0],       # p(0)
        [0, 1, 0,    0,     0,      0],       # p'(0)
        [0, 0, 2,    0,     0,      0],       # p''(0)
        [1, T, T2,   T3,    T4,     T5],      # p(T)
        [0, 1, 2*T,  3*T2,  4*T3,   5*T4],    # p'(T)
        [0, 0, 2,    6*T,   12*T2,  20*T3],   # p''(T)
    ], dtype=float)


def _jerk_cost_matrix(T):
    """W (6x6) with cost = coeffs^T W coeffs = integral_0^T (p''')^2 dt.
    Only the c3,c4,c5 block is non-zero."""
    W = np.zeros((6, 6))
    W[3, 3] = 36.0*T
    W[3, 4] = W[4, 3] = 72.0*T**2
    W[3, 5] = W[5, 3] = 120.0*T**3
    W[4, 4] = 192.0*T**3
    W[4, 5] = W[5, 4] = 360.0*T**4
    W[5, 5] = 720.0*T**5
    return W


class _Quintic:
    """A concrete 1D quintic on [0, T]. Vectorized eval."""
    def __init__(self, T, bc):
        self.T = float(T)
        self.coeffs = np.linalg.solve(_bc_matrix(self.T), np.asarray(bc, dtype=float))

    def _pow(self, t):
        t = np.asarray(t, dtype=float)
        return np.stack([np.ones_like(t), t, t**2, t**3, t**4, t**5], axis=-1)

    def pos(self, t):
        return self._pow(t) @ self.coeffs

    def vel(self, t):
        t = np.asarray(t, dtype=float)
        d = np.stack([np.zeros_like(t), np.ones_like(t), 2*t, 3*t**2, 4*t**3, 5*t**4], axis=-1)
        return d @ self.coeffs

    def acc(self, t):
        t = np.asarray(t, dtype=float)
        d = np.stack([np.zeros_like(t), np.zeros_like(t), 2*np.ones_like(t),
                      6*t, 12*t**2, 20*t**3], axis=-1)
        return d @ self.coeffs


class _Straight:
    """1D quintic in arc length, embedded along a fixed unit direction."""
    def __init__(self, T, bc1d, p_start, direction):
        self.T = float(T)
        self.q = _Quintic(T, bc1d)
        self.p_start = np.asarray(p_start, dtype=float)
        self.dir = np.asarray(direction, dtype=float)

    def eval(self, t):
        s = self.q.pos(t)[..., None]
        v = self.q.vel(t)[..., None]
        a = self.q.acc(t)[..., None]
        return self.p_start + s*self.dir, v*self.dir, a*self.dir


class _Blend:
    """Per-axis quintics over [0, T]."""
    def __init__(self, T, bc_per_axis):
        self.T = float(T)
        self.qs = [_Quintic(T, bc) for bc in bc_per_axis]

    def eval(self, t):
        pos = np.stack([q.pos(t) for q in self.qs], axis=-1)
        vel = np.stack([q.vel(t) for q in self.qs], axis=-1)
        acc = np.stack([q.acc(t) for q in self.qs], axis=-1)
        return pos, vel, acc


class Trajectory:
    """Time-parameterized trajectory. sample(t) -> (pos, vel, acc); .duration."""
    def __init__(self, pieces, dim):
        self.pieces = pieces                       # ordered list of path pieces
        self.dim = dim
        self.alpha = 1.0                           # global uniform time scale
        self._base_T = np.array([p.T for p in pieces], dtype=float)
        self._cum = np.concatenate([[0.0], np.cumsum(self._base_T)])

    @property
    def base_duration(self):
        return float(self._cum[-1])

    @property
    def duration(self):
        if(self.alpha < 1):
            self.alpha = 1
        return self.alpha * self.base_duration

    def sample(self, t):
        """Scalar or array t -> (pos, vel, acc). Applies alpha time-scaling:
        pos unchanged, vel ~ 1/alpha, acc ~ 1/alpha^2."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        tau = np.clip(t/self.alpha, 0.0, self.base_duration)
        pos = np.zeros((len(tau), self.dim))
        vel = np.zeros((len(tau), self.dim))
        acc = np.zeros((len(tau), self.dim))
        for i, piece in enumerate(self.pieces):
            lo, hi = self._cum[i], self._cum[i+1]
            mask = (tau >= lo) & (tau <= hi) if i == len(self.pieces)-1 else (tau >= lo) & (tau < hi)
            if not mask.any():
                continue
            local = tau[mask] - lo
            p, v, a = piece.eval(local)
            pos[mask], vel[mask], acc[mask] = p, v/self.alpha, a/(self.alpha**2)
        return pos, vel, acc

    def sample_grid(self, n):
        ts = np.linspace(0.0, self.duration, n)
        pos, vel, acc = self.sample(ts)
        return ts, pos, vel, acc


# ============================================================================
# GEOMETRY: corners, cut points, tangents
# ============================================================================
def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _build_geometry(W, f):
    """W: (n, d) ordered start->goal. Returns straights' trimmed endpoints,
    per-corner cut points + tangents, and segment lengths."""
    n = len(W)
    tang = [_unit(W[i+1]-W[i]) for i in range(n-1)]        # straight tangents
    L    = [np.linalg.norm(W[i+1]-W[i]) for i in range(n-1)]
    m    = n - 2                                           # interior corners

    d = np.zeros(max(m, 0))
    for k in range(m):
        d[k] = f * min(L[k], L[k+1])
    '''
    # defensive clamp: no straight may have start-cut + end-cut exceed its length
    for s in range(n-1):
        d_start = d[s-1] if s >= 1 else 0.0
        d_end   = d[s]   if s <= n-3 else 0.0
        if d_start + d_end > L[s] - 1e-9:
            scale = max(1e-9, (L[s] - 1e-6)) / (d_start + d_end)
            if s >= 1:    d[s-1] *= scale
            if s <= n-3:  d[s] *= scale
'''
    corners = []
    for k in range(m):
        vk = k + 1
        u_in, u_out = tang[k], tang[k+1]
        c_in  = W[vk] - d[k]*u_in
        c_out = W[vk] + d[k]*u_out
        corners.append(dict(vk=vk, u_in=u_in, u_out=u_out, c_in=c_in, c_out=c_out))

    # trimmed straight endpoints
    straights = []
    for s in range(n-1):
        p0 = W[s]   + (d[s-1]*tang[s] if s >= 1     else 0.0)
        p1 = W[s+1] - (d[s]  *tang[s] if s <= n-3   else 0.0)
        straights.append(dict(p0=p0, p1=p1, dir=tang[s],
                              L=float(np.linalg.norm(p1-p0))))
    return straights, corners, tang, L, d


# ============================================================================
# QP ASSEMBLY + SOLVE (single global minimum-jerk problem)
# ============================================================================
def generate_trajectory(waypoints, vmax, amax,f,env=None, verbose=True):
    """Build a smooth, dynamically bounded min-jerk trajectory over waypoints.

    waypoints : (n, d) array-like, ORDERED start -> goal, already obstacle-inflated.
    vmax, amax: dynamic limits enforced by uniform post-scaling.
    env       : optional Environment3D for collision check + reactive insertion.
    Returns   : Trajectory (sample(t) -> pos,vel,acc ; .duration).
    """

    W = np.array(waypoints, dtype=float)
    if W.ndim != 2 or len(W) < 2:
        raise ValueError("waypoints must be (n>=2, d)")
    dim = W.shape[1]

    for it in range(MAX_REPAIR_ITERS + 1):
        straights, corners, _, _, _ = _build_geometry(W, f)
        x = _solve_min_jerk(W, straights, corners, dim)
        pieces = _build_pieces(W, straights, corners, x, dim)
        traj = Trajectory(pieces, dim)
        v_peak, a_peak = _apply_dynamic_limits(traj, vmax, amax)

        if env is None:
            if verbose:
                print(f"[traj] no env -> skipped collision check. "
                      f"dur={traj.duration:.2f}s v_peak={v_peak:.2f} a_peak={a_peak:.2f}")
            return traj

        bad = _first_collision(traj, env)
        if bad is None:
            if verbose:
                print(f"[traj] collision-free after {it} repair(s). "
                      f"waypoints={len(W)} dur={traj.duration:.2f}s "
                      f"v_peak={v_peak:.2f} a_peak={a_peak:.2f}")
            return traj

        newW = _repair_waypoints(W, corners, bad)
        if newW is None or len(newW) == len(W):
            raise RuntimeError(f"[traj] corner too tight near piece {bad}; "
                               f"cannot repair (iter {it}).")
        if verbose:
            print(f"[traj] collision at piece {bad}; inserting waypoint "
                  f"(iter {it+1}), {len(W)} -> {len(newW)} waypoints.")
        W = newW

    raise RuntimeError(f"[traj] exceeded MAX_REPAIR_ITERS={MAX_REPAIR_ITERS}; "
                       f"corner too tight.")






def _solve_min_jerk(W, straights, corners, dim):
    """Decision vars per corner k: [v_in, a_in, v_out, a_out] at base index 4k.
    Returns solved magnitude vector x (len 4*m)."""
    n = len(W)
    m = len(corners)
    nx = 4*m
    if nx == 0:
        return np.zeros(0)

    H = np.zeros((nx, nx))
    fvec = np.zeros(nx)

    def accumulate(S, c, T):
        Minv = np.linalg.inv(_bc_matrix(T))
        Q = Minv.T @ _jerk_cost_matrix(T) @ Minv
        nonlocal H, fvec
        H += S.T @ Q @ S
        fvec += S.T @ Q @ c

    # --- straights (1D) ---
    for s in range(n-1):
        st = straights[s]
        T = max(st['L']/V_AVG, 1e-3)
        S = np.zeros((6, nx)); c = np.zeros(6)
        c[0] = 0.0; c[3] = st['L']
        if s >= 1:                       # start junction = c_out of corner s-1
            S[1, 4*(s-1)+2] = 1.0; S[2, 4*(s-1)+3] = 1.0
        if s <= n-3:                     # end junction = c_in of corner s
            S[4, 4*s+0] = 1.0; S[5, 4*s+1] = 1.0
        accumulate(S, c, T)

    # --- blends (per axis) ---
    for k, cor in enumerate(corners):
        T = max(np.linalg.norm(cor['c_out']-cor['c_in'])/V_AVG, 1e-3)
        for j in range(dim):
            S = np.zeros((6, nx)); c = np.zeros(6)
            c[0] = cor['c_in'][j]; c[3] = cor['c_out'][j]
            S[1, 4*k+0] = cor['u_in'][j]
            S[2, 4*k+1] = cor['u_in'][j]
            S[4, 4*k+2] = cor['u_out'][j]
            S[5, 4*k+3] = cor['u_out'][j]
            accumulate(S, c, T)

    x = np.linalg.solve(H + _REG*np.eye(nx), -fvec)   # minimize x^T H x + 2 f^T x
    return x


def _build_pieces(W, straights, corners, x, dim):
    """Instantiate concrete path pieces in path order: S0,B0,S1,B1,...,S_{n-2}."""
    n = len(W)
    pieces = []
    for s in range(n-1):
        st = straights[s]
        T = max(st['L']/V_AVG, 1e-3)
        v0 = x[4*(s-1)+2] if s >= 1   else 0.0
        a0 = x[4*(s-1)+3] if s >= 1   else 0.0
        v1 = x[4*s+0]     if s <= n-3 else 0.0
        a1 = x[4*s+1]     if s <= n-3 else 0.0
        bc = [0.0, v0, a0, st['L'], v1, a1]
        pieces.append(_Straight(T, bc, st['p0'], st['dir']))
        if s <= n-3:                                   # blend after straight s
            cor = corners[s]
            T_b = max(np.linalg.norm(cor['c_out']-cor['c_in'])/V_AVG, 1e-3)
            bc_axes = []
            for j in range(dim):
                bc_axes.append([cor['c_in'][j],
                                x[4*s+0]*cor['u_in'][j],  x[4*s+1]*cor['u_in'][j],
                                cor['c_out'][j],
                                x[4*s+2]*cor['u_out'][j], x[4*s+3]*cor['u_out'][j]])
            pieces.append(_Blend(T_b, bc_axes))
    return pieces


# ============================================================================
# TIME SCALING (uniform, after solve)
# ============================================================================
def _apply_dynamic_limits(traj, vmax, amax, n=2000):
    """alpha = max(v_peak/vmax, sqrt(a_peak/amax)); scale ALL times once."""
    ts = np.linspace(0.0, traj.base_duration, n)
    _, vel, acc = traj.sample(ts)                      # alpha == 1 here
    v_peak = float(np.max(np.linalg.norm(vel, axis=1))) if len(vel) else 0.0
    a_peak = float(np.max(np.linalg.norm(acc, axis=1))) if len(acc) else 0.0
    alpha = max(v_peak/vmax if vmax > 0 else 0.0,
                np.sqrt(a_peak/amax) if amax > 0 else 0.0)
    traj.alpha = alpha if alpha > 1e-9 else 1.0
    return v_peak, a_peak


# ============================================================================
# COLLISION CHECK + REACTIVE INSERTION
# ============================================================================
def _first_collision(traj, env):
    """Dense arc-length sampling; returns index of colliding piece, else None."""
    total_len = sum(p.T*V_AVG for p in traj.pieces) + 1e-6   # base-time length est.
    n = max(50, int(total_len / (COLLISION_STEP*0.5)))
    tau = np.linspace(0.0, traj.base_duration, n)
    # sample in BASE time (alpha=1) so tau maps straight back to piece boundaries
    saved, traj.alpha = traj.alpha, 1.0
    pos, _, _ = traj.sample(tau)
    traj.alpha = saved

    if env.is_point_in_free_space(pos):           # batch: True iff ALL free
        return None
    for i in range(len(pos)):                     # locate first offending sample
        if not env.is_point_in_free_space(pos[i]):
            idx = int(np.searchsorted(traj._cum, tau[i], side='right') - 1)
            return int(np.clip(idx, 0, len(traj.pieces)-1))
    return None


def _repair_waypoints(W, corners, piece_idx):
    """Insert safe corridor points near the offending corner to shrink its cut.
    Pieces are ordered S0,B0,S1,B1,... -> blend index = (piece_idx-1)//2."""
    m = len(corners)
    if m == 0:
        return None
    if piece_idx % 2 == 1:                      # a blend piece
        k = (piece_idx - 1)//2
    else:                                       # a straight; blame nearest corner
        s = piece_idx // 2
        k = min(max(s-1, 0), m-1) if s >= 1 else 0
    cor = corners[k]
    vk = cor['vk']
    d_half_in  = 0.5*np.linalg.norm(cor['c_in']  - W[vk])
    d_half_out = 0.5*np.linalg.norm(cor['c_out'] - W[vk])
    q_in  = W[vk] - d_half_in  * cor['u_in']
    q_out = W[vk] + d_half_out * cor['u_out']
    newW = list(W[:vk])
    if np.linalg.norm(q_in - W[vk]) > 1e-3:
        newW.append(q_in)
    newW.append(W[vk])
    if np.linalg.norm(q_out - W[vk]) > 1e-3:
        newW.append(q_out)
    newW.extend(W[vk+1:])
    return np.array(newW, dtype=float)




# ============================================================================
# DOWNSTREAM ADAPTER (arrays for control.QuadrotorController.set_trajectory)
# ============================================================================
def sample_arrays(traj, dt=0.02):
    """Sample -> (points, time_points, velocities, accelerations) at fixed dt."""
    n = max(2, int(round(traj.duration/dt)) + 1)
    ts, pos, vel, acc = traj.sample_grid(n)
    return pos, ts, vel, acc


# ============================================================================
# PLOTTING / INSPECTION
# ============================================================================
def plot_diagnostics(traj, n=1500, savepath=None, show=True):
    """Curvature vs arc length, and speed & acceleration vs time."""
    ts, pos, vel, acc = traj.sample_grid(n)
    speed = np.linalg.norm(vel, axis=1)
    acc_mag = np.linalg.norm(acc, axis=1)

    ds = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(ds)])

    cross = np.cross(vel, acc)
    num = np.linalg.norm(cross, axis=1) if cross.ndim > 1 else np.abs(cross)
    denom = np.clip(speed**3, 1e-9, None)
    curvature = num/denom

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(arc, curvature, 'b-', lw=1.5)
    a1.set_xlabel('Arc length (m)'); a1.set_ylabel('Curvature (1/m)')
    a1.set_title('Curvature vs Arc Length'); a1.grid(True)

    a2.plot(ts, speed,   'g-', lw=1.5, label='|v| (m/s)')
    a2.plot(ts, acc_mag, 'm-', lw=1.5, label='|a| (m/s^2)')
    a2.set_xlabel('Time (s)'); a2.set_ylabel('Magnitude')
    a2.set_title('Speed & Acceleration vs Time'); a2.legend(); a2.grid(True)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=110)
    if show:
        plt.show()
    return fig
