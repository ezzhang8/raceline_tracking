import numpy as np
import math
from numpy.typing import ArrayLike

from simulator import RaceTrack


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _curvature_three_points(p_prev: np.ndarray,
                            p_curr: np.ndarray,
                            p_next: np.ndarray) -> float:
    a = np.linalg.norm(p_curr - p_prev)
    b = np.linalg.norm(p_next - p_curr)
    c = np.linalg.norm(p_next - p_prev)

    eps = 1e-6
    if a < eps or b < eps or c < eps:
        return 0.0

    area = 0.5 * abs(np.cross(p_curr - p_prev, p_next - p_prev))
    if area < eps:
        return 0.0

    R = (a * b * c) / (4.0 * area)
    if R < eps:
        return 0.0

    return 1.0 / R


def _scan_corner_ahead(path: np.ndarray,
                       idx_start: int,
                       kappa_thresh: float,
                       s_max_scan: float) -> tuple[float, float | None]:
    N = path.shape[0]
    s_acc = 0.0
    kappa_max = 0.0
    s_corner = None

    idx = idx_start
    for _ in range(N):
        idx_prev = (idx - 1) % N
        idx_next = (idx + 1) % N

        p_prev = path[idx_prev]
        p_curr = path[idx]
        p_next = path[idx_next]

        kappa_here = _curvature_three_points(p_prev, p_curr, p_next)
        kappa_max = max(kappa_max, kappa_here)

        if kappa_here >= kappa_thresh and s_corner is None:
            s_corner = s_acc

        idx_fwd = (idx + 1) % N
        ds = float(np.linalg.norm(path[idx_fwd] - p_curr))
        s_acc += ds
        if s_acc >= s_max_scan:
            break
        idx = idx_fwd

    return kappa_max, s_corner


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> np.ndarray:
    state = np.asarray(state, float)
    parameters = np.asarray(parameters, float)

    sx, sy, delta, v, phi = state
    L         = float(parameters[0])
    delta_min = float(parameters[1])
    v_min     = float(parameters[2])
    delta_max = float(parameters[4])
    v_max     = float(parameters[5])

    path = np.asarray(racetrack.centerline, float)
    N = path.shape[0]

    front_pos = np.array([
        sx + 2.0 * L * math.cos(phi),
        sy + 2.0 * L * math.sin(phi),
    ])

    diffs = path - front_pos[None, :]
    dists2 = np.einsum("ij,ij->i", diffs, diffs)
    idx_min = int(np.argmin(dists2))

    idx_prev = (idx_min - 1) % N
    idx_next = (idx_min + 1) % N

    p_prev = path[idx_prev]
    p_curr = path[idx_min]
    p_next = path[idx_next]

    kappa_local = _curvature_three_points(p_prev, p_curr, p_next)


    KAPPA_TURN     = 0.01
    LOOKAHEAD_ARC  = 120.0
    kappa_ahead, s_corner = _scan_corner_ahead(
        path, idx_min, KAPPA_TURN, LOOKAHEAD_ARC
    )
    effective_kappa = max(kappa_local, kappa_ahead)

    if effective_kappa > 0.02:
        LA = 2
    elif effective_kappa > 0.01:
        LA = 3
    elif effective_kappa > 0.005:
        LA = 5
    else:
        LA = 9 if v > 50.0 else (3 if v > 30.0 else 2)

    target = path[(idx_min + LA) % N]

    dx = float(target[0] - sx)
    dy = float(target[1] - sy)

    cp, sp = math.cos(phi), math.sin(phi)
    x_local = cp * dx + sp * dy
    y_local = -sp * dx + cp * dy

    if x_local <= 0.1:
        delta_des = delta
    else:
        Ld2 = x_local * x_local + y_local * y_local
        curvature_cmd = 2.0 * y_local / max(Ld2, 1e-6)
        delta_des = math.atan(L * curvature_cmd)

    delta_des = float(np.clip(delta_des, delta_min, delta_max))

    A_LAT_MAX = 30
    eps_k = 1e-4
    if effective_kappa < eps_k:
        v_curve_limit = 1e6
    else:
        v_curve_limit = math.sqrt(A_LAT_MAX / (effective_kappa + eps_k))

    V_MAX_STRAIGHT = 0.7 * v_max

    v_base = min(V_MAX_STRAIGHT, v_curve_limit)

    A_BRAKE_PLAN = 0.4

    if s_corner is None:
        v_des = v_base
    else:
        if kappa_ahead < eps_k:
            v_corner_safe = V_MAX_STRAIGHT
        else:
            v_corner_safe = math.sqrt(A_LAT_MAX / (kappa_ahead + eps_k))

        v_brake_based = math.sqrt(
            max(v_corner_safe * v_corner_safe + 2.0 * A_BRAKE_PLAN * s_corner, 0.0)
        )

        v_des = min(v_base, v_brake_based)
    
    v_des = float(np.clip(v_des, v_min, v_max))

    steer_ratio = abs(delta_des) / max(1e-6, delta_max)
    if steer_ratio > 0.4: 
        v_des = min(v_des, 0.1 * v_max)
    elif steer_ratio > 0.2: 
        v_des = min(v_des, 0.15 * v_max)
    elif steer_ratio > 0.1:
        v_des = min(v_des, 0.2 * v_max)


    return np.array([delta_des, v_des], dtype=float)


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    state = np.asarray(state, dtype=float)
    desired = np.asarray(desired, dtype=float)
    parameters = np.asarray(parameters, dtype=float)

    delta = state[2]
    v = state[3]
    delta_des, v_des = desired

    delta_dot_min = parameters[7]
    a_min         = parameters[8]
    delta_dot_max = parameters[9]
    a_max         = parameters[10]

    k_delta_p = 6
    delta_err = _wrap_angle(delta_des - delta)
    v_delta_cmd = k_delta_p * delta_err

    k_v_p = 1000
    a_cmd = k_v_p * (v_des - v)

    v_delta_cmd = float(np.clip(v_delta_cmd, delta_dot_min, delta_dot_max))
    a_cmd       = float(np.clip(a_cmd,       a_min,         a_max))

    return np.array([v_delta_cmd, a_cmd], dtype=float)
