# baja_pure_pursuit_cones.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon

# -----------------------
# Config
# -----------------------
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

DT = 0.05                # time step [s]
TOTAL_TIME = 60.0        # total sim time [s]
WHEELBASE = 2.4          # vehicle wheelbase [m]
MAX_STEER = np.deg2rad(30)  # steering limit [rad]
LOOKAHEAD_MIN = 3.0      # min lookahead [m]
LOOKAHEAD_MAX = 8.0      # max lookahead [m]
LOOKAHEAD_GAIN = 0.3     # ld = clip(v * gain, min, max)
TARGET_SPEED = 6.0       # [m/s]
TRACK_WIDTH = 5.0        # [m]
CONES_EVERY = 10         # place cones every N centerline samples
ANIM_FPS = 20

# -----------------------
# Utilities
# -----------------------
def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n

def triangle_at(x, y, yaw, length=2.0, width=1.0):
    pts = np.array([[ length/2, 0.0],
                    [-length/2, -width/2],
                    [-length/2,  width/2]])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s],[s, c]])
    pts = (R @ pts.T).T + np.array([x, y])
    return pts

# -----------------------
# Track generation (figure-8 with cones)
# -----------------------
def generate_track(num=800, scale=20.0, width=TRACK_WIDTH):
    t = np.linspace(0, 2*np.pi, num, endpoint=False)
    cx = scale * np.sin(t)
    cy = scale * np.sin(t) * np.cos(t)
    center = np.vstack((cx, cy)).T
    tangents = np.gradient(center, axis=0)
    normals = np.zeros_like(tangents)
    for i in range(num):
        tx, ty = tangents[i]
        n = unit(np.array([-ty, tx]))
        normals[i] = n
    left = center + (width/2.0) * normals
    right = center - (width/2.0) * normals
    left_cones = left[::CONES_EVERY]
    right_cones = right[::CONES_EVERY]
    pairs = min(len(left_cones), len(right_cones))
    center_wp = (left_cones[:pairs] + right_cones[:pairs]) / 2.0
    return center_wp, left_cones, right_cones

def nearest_index(path, x, y, start_idx=0, window=50):
    N = len(path)
    idxs = [(start_idx + i) % N for i in range(-window, window+1)]
    pts = path[idxs]
    d = np.hypot(pts[:,0]-x, pts[:,1]-y)
    k = np.argmin(d)
    return idxs[k]

def lookahead_point(path, x, y, start_idx, ld):
    # walk forward along path until accumulated length >= ld
    N = len(path)
    idx = nearest_index(path, x, y, start_idx=start_idx, window=30)
    acc = 0.0
    while acc < ld:
        p0 = path[idx % N]
        p1 = path[(idx+1) % N]
        step = np.hypot(*(p1 - p0))
        acc += step
        idx = (idx + 1) % N
        # safety break
        if acc > 10*ld:
            break
    return path[idx % N], idx % N

def curvature_heading(path, idx):
    N = len(path)
    p_next = path[(idx+1) % N]
    p = path[idx % N]
    v = p_next - p
    return np.arctan2(v[1], v[0])

# -----------------------
# Pure Pursuit Controller
# -----------------------
class PurePursuit:
    def __init__(self, wheelbase=WHEELBASE):
        self.L = wheelbase
        self.last_idx = 0

    def compute(self, state, path, v):
        x, y, yaw = state
        ld = np.clip(v * LOOKAHEAD_GAIN, LOOKAHEAD_MIN, LOOKAHEAD_MAX)
        target, self.last_idx = lookahead_point(path, x, y, self.last_idx, ld)
        dx, dy = target[0] - x, target[1] - y
        c, s = np.cos(-yaw), np.sin(-yaw)
        local_x = c*dx - s*dy
        local_y = s*dx + c*dy
        alpha = np.arctan2(local_y, local_x)
        delta = np.arctan2(2*self.L*np.sin(alpha)/ld, 1.0)
        delta = np.clip(delta, -MAX_STEER, MAX_STEER)
        return delta, target, ld

# -----------------------
# Simulation
# -----------------------
def run_sim():
    path, left_cones, right_cones = generate_track()
    x, y = path[0]
    yaw = curvature_heading(path, 0)
    v = TARGET_SPEED
    state = np.array([x, y, yaw], dtype=float)

    ctrl = PurePursuit(WHEELBASE)

    ts = []
    xs, ys, yaws = [], [], []
    speeds, deltas = [], []
    lat_errs, head_errs = [], []

    steps = int(TOTAL_TIME / DT)
    for step in range(steps):
        t = step * DT
        delta, target, ld = ctrl.compute(state, path, v)

        x, y, yaw = state
        x += v * np.cos(yaw) * DT
        y += v * np.sin(yaw) * DT
        yaw += v / WHEELBASE * np.tan(delta) * DT
        yaw = wrap_angle(yaw)
        state = np.array([x, y, yaw])

        ts.append(t)
        xs.append(x); ys.append(y); yaws.append(yaw)
        speeds.append(v); deltas.append(delta)

        idx = nearest_index(path, x, y, ctrl.last_idx, window=20)
        p = path[idx]
        heading_ref = curvature_heading(path, idx)
        e_vec = np.array([x - p[0], y - p[1]])
        t_vec = np.array([np.cos(heading_ref), np.sin(heading_ref)])
        n_vec = np.array([-t_vec[1], t_vec[0]])
        lat_err = np.dot(e_vec, n_vec)
        head_err = wrap_angle(yaw - heading_ref)
        lat_errs.append(lat_err); head_errs.append(head_err)

    logs = {
        "t": np.array(ts),
        "x": np.array(xs),
        "y": np.array(ys),
        "yaw": np.array(yaws),
        "v": np.array(speeds),
        "delta": np.array(deltas),
        "lat_err": np.array(lat_errs),
        "head_err": np.array(head_errs),
        "path": path,
        "left_cones": left_cones,
        "right_cones": right_cones
    }
    return logs

# -----------------------
# Robust animation & telemetry
# -----------------------
def make_animation(logs, fname=os.path.join(OUT_DIR, "run.gif")):
    path = logs["path"]
    left_cones = logs["left_cones"]
    right_cones = logs["right_cones"]
    xs = logs["x"]
    ys = logs["y"]
    yaws = logs["yaw"]

    frames = len(xs)
    if frames == 0:
        print("No trajectory data to animate.")
        return

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_aspect('equal', 'box')
    ax.scatter(left_cones[:,0], left_cones[:,1], marker='o', s=40, label='Left cones')
    ax.scatter(right_cones[:,0], right_cones[:,1], marker='^', s=40, label='Right cones')
    ax.plot(path[:,0], path[:,1], '--', linewidth=1, label='Centerline')
    car_poly = Polygon(triangle_at(xs[0], ys[0], yaws[0]), closed=True, alpha=0.8)
    ax.add_patch(car_poly)
    car_traj, = ax.plot([], [], '-', linewidth=1.5, label='Trajectory')
    steer_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')

    all_x = np.concatenate([path[:,0], left_cones[:,0], right_cones[:,0], xs])
    all_y = np.concatenate([path[:,1], left_cones[:,1], right_cones[:,1], ys])
    margin = 5.0
    ax.set_xlim(all_x.min()-margin, all_x.max()+margin)
    ax.set_ylim(all_y.min()-margin, all_y.max()+margin)
    ax.set_title("BAJA Pure Pursuit on Cone-Defined Track")
    ax.legend(loc='lower right')

    def init():
        car_traj.set_data([], [])
        steer_text.set_text('')
        return car_poly, car_traj, steer_text

    def update(frame):
        idx = min(frame, frames-1)
        x = xs[idx]; y = ys[idx]; yaw = yaws[idx]
        car_poly.set_xy(triangle_at(x, y, yaw, length=2.0, width=1.2))
        car_traj.set_data(xs[:idx+1], ys[:idx+1])
        steer_text.set_text(f"Frame: {idx+1}/{frames}")
        return car_poly, car_traj, steer_text

    ani = animation.FuncAnimation(fig, update, init_func=init, frames=frames,
                                  interval=1000/ANIM_FPS, blit=False)

    try:
        from matplotlib.animation import PillowWriter
        ani.save(fname, writer=PillowWriter(fps=ANIM_FPS))
        print(f"Saved animation to {fname}")
    except Exception as e:
        print("Failed to save GIF:", repr(e))
    plt.close(fig)

def plot_telemetry(logs, fname=os.path.join(OUT_DIR, "telemetry.png")):
    t = logs.get("t", np.array([]))
    if t.size == 0:
        print("No telemetry data to plot.")
        return

    delta_deg = np.rad2deg(logs["delta"])

    fig, axs = plt.subplots(3, 1, figsize=(9,8), sharex=True)
    axs[0].plot(t, logs["v"]); axs[0].set_ylabel("Speed [m/s]")
    axs[1].plot(t, delta_deg); axs[1].set_ylabel("Steer [deg]")
    axs[2].plot(t, logs["lat_err"]); axs[2].set_ylabel("Lat error [m]")
    axs[2].set_xlabel("Time [s]")
    for ax in axs: ax.grid(True)
    fig.suptitle("Telemetry")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved telemetry to {fname}")

if __name__ == "__main__":
    logs = run_sim()
    print("Simulation finished. Frames generated:", len(logs.get("x", [])))
    make_animation(logs)
    plot_telemetry(logs)
