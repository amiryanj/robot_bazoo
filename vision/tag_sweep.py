#!/usr/bin/env python
"""Measure the wrist_roll mapping offset DELTA with the ArUco finger tags.

A tag rigid w.r.t. the gripper body can NOT observe DELTA: FK(roll+d) =
FK(roll) ∘ C(d) with C(d) constant, so any d is absorbed by the unknown tag
mounting transform — a pure gauge freedom (the 2026-06-12 selftest proved it:
data generated at delta=+13.5deg fit perfectly at -3.2deg). The fix uses the
one DRIVEN jaw of the SO-101 (`moving_jaw_so101_v1`): sweeping the gripper
opening swings that finger's tag about the JAW axis, which is not parallel to
the roll axis, and the direction the tag travels betrays the true roll
orientation. No assumption about how the tags are glued is needed.

Unknowns: DELTA (1) + jaw-map scale A (jaw_deg = A*reading + B0; B0 stays
frozen — an offset in B is the same gauge freedom against the jaw tag's
mount; A is seeded at both signs) + a 6-DoF mounting transform per tag.
Which tag rides the moving jaw is decided by the data: solve once per
hypothesis, keep the lowest rms.

Hard-won details (first run scored 10deg/25mm rms before these):
- samples store raw CORNERS + intrinsics, not poses, so PnP can be redone
  offline (--solve) with different choices;
- tag side is the known 24 mm print, NOT depth-measured — at the safe sweep
  height the fingers sit inside the D455 min-Z blind zone, and an evolving
  median made the scale inconsistent across the run;
- planar PnP (IPPE_SQUARE) has a two-fold ambiguity at small tags: both
  candidate poses are kept per sample; the solver alternates between fitting
  (with the assignment frozen, soft_l1 robust loss) and re-assigning each
  sample to the candidate the current model explains better, seeded by
  reprojection error. A naive min-inside-the-residual biased delta by ~1deg
  even on clean synthetic data; position-only residuals lose observability
  (several-degree wander) — rotations carry the delta signal.

Safety: every pose is FK-checked (tight geom AABBs, both gripper extremes)
to keep the fingers >= 5 cm above the plate; unsafe poses are skipped.
Live view: detections + solver results stream to Rerun (spawned viewer);
the solver also saves a residual figure next to the calib JSON.

    python vision/tag_sweep.py [--port /dev/ttyACM1]
    python vision/tag_sweep.py --solve outputs/calib/tag_sweep_samples.json
    python vision/tag_sweep.py --selftest
"""
import argparse
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

from config import ARM_PORT   # noqa: E402

# raised vs the first sweep (lift 22 -> 0, elbow 45 -> 35): keeps the fingers
# >= 5 cm above the plate at every pose incl. gripper extremes (AABB-checked).
BASE_POSE = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 35.0,
             "wrist_flex": -40.0, "wrist_roll": 0.0, "gripper": 20.0}
# (pan, wf, rolls): pan MUST vary — with pan fixed the roll axis is one line in
# space and "rotate the camera about that line" vs "shift DELTA" is a near-exact
# gauge once extrinsics are co-fitted (run 2 drifted 17deg/82mm chasing it).
ROLL_SWEEPS = [(0.0, -40.0, np.arange(-100, 101, 10.0)),
               (0.0, -60.0, np.arange(-60, 41, 20.0)),
               (0.0, -20.0, np.arange(-60, 41, 20.0)),
               (-25.0, -40.0, np.arange(-60, 41, 20.0)),
               (25.0, -40.0, np.arange(-60, 41, 20.0))]
GRIPPER_LEVELS = (5.0, 14.0, 23.0, 32.0, 41.0, 50.0, 59.0)   # makes DELTA observable
N_GRIP_POSES = 4   # one gripper block per distinct (pan, wf) — axis-direction variety
MOTOR_NAMES = list(BASE_POSE)
W_POS = 10.0                       # 10 cm position error ~ 1 rad orientation error
VALID_IDS = (1, 2)                 # one tag per finger
TAG_SIDE = 0.024                   # the print size; constant scale across the whole run
JAW_BODY, GRIP_BODY = "moving_jaw_so101_v1", "gripper"
A0, B0 = 1.1, -10.0                # jaw_deg = A*reading + B0 prior (joint range -10..100deg)
PLATE_Z = -0.029                   # plate height in base frame (CLAUDE.md geometry)
MIN_CLEAR = 0.06                   # 5 cm user rule + 1 cm sticker margin


def make_fk2():
    """fk(ang, delta_deg, a) -> {body: (R, t)} for the gripper body and the moving
    jaw (jaw qpos = radians(a*gripper_reading + B0)), plus lowest_z(ang): the
    lowest finger-geom AABB corner over both gripper extremes (collision check)."""
    import mujoco
    from realsense import XML
    mm = mujoco.MjModel.from_xml_path(XML)
    md = mujoco.MjData(mm)
    adr = {j: mm.jnt_qposadr[mm.joint(j).id] for j in MOTOR_NAMES}
    bids = {b: mm.body(b).id for b in (GRIP_BODY, JAW_BODY)}
    fingers = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] in bids.values()]

    def set_q(ang, delta, a):
        for j, qa in adr.items():
            md.qpos[qa] = math.radians(ang[j])
        md.qpos[adr["wrist_roll"]] = math.radians(ang["wrist_roll"] + delta)
        md.qpos[adr["gripper"]] = math.radians(a * ang["gripper"] + B0)
        mujoco.mj_forward(mm, md)

    def fk(ang, delta=0.0, a=A0):
        set_q(ang, delta, a)
        return {b: (md.xmat[i].reshape(3, 3).copy(), md.xpos[i].copy())
                for b, i in bids.items()}

    def lowest_z(ang):
        lo = 1e9
        for g_ext in (2.0, 60.0):
            set_q(dict(ang, gripper=g_ext), 0.0, A0)
            for g in fingers:
                c, s = mm.geom_aabb[g][:3], mm.geom_aabb[g][3:]
                R, p = md.geom_xmat[g].reshape(3, 3), md.geom_xpos[g]
                lo = min(lo, min((p + R @ (c + np.array(sgn) * s))[2]
                                 for sgn in itertools.product((-1, 1), repeat=3)))
        return lo

    return fk, lowest_z


def detect_tags(det_aruco, color):
    """All valid tags in the frame -> list of (id, corners 4x2)."""
    import cv2
    corners, ids, _ = det_aruco.detectMarkers(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return []
    return [(int(tid), c4[0]) for c4, tid in zip(corners, ids.ravel())
            if int(tid) in VALID_IDS]


def pnp_candidates(corners, K):
    """Both IPPE_SQUARE solutions (planar two-fold ambiguity) ->
    [(R_ct, t_ct, reproj_err), ...] sorted by reprojection error."""
    import cv2
    S = TAG_SIDE
    obj = np.array([[-S / 2, S / 2, 0], [S / 2, S / 2, 0],
                    [S / 2, -S / 2, 0], [-S / 2, -S / 2, 0]], np.float32)
    Km = np.array([[K["fx"], 0, K["ppx"]], [0, K["fy"], K["ppy"]], [0, 0, 1]], np.float64)
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj, np.asarray(corners, np.float32), Km, np.zeros(5),
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    errs = np.ravel(errs)
    out = [(cv2.Rodrigues(rvecs[i])[0], tvecs[i].ravel(), float(errs[i]))
           for i in range(n)]
    return sorted(out, key=lambda c: c[2])


def load_handeye():
    import json
    he = json.load(open(ROOT / "outputs/calib/handeye.json"))
    return np.array(he["R"]), np.array(he["t"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=ARM_PORT)
    ap.add_argument("--solve", metavar="JSON",
                    help="skip the sweep: load saved samples and re-run the solver")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic end-to-end solver check (no hardware)")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import json
    if args.solve:
        samples = [(s["tid"], s["ang"], np.array(s["corners"]), s["K"])
                   for s in json.load(open(args.solve))]
        solve(samples)
        return

    import cv2
    import rerun as rr
    from realsense import Realsense
    from pick_ball import move_to, read_angles
    from gamepad_utils import graceful_shutdown
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    rr.init("tag_sweep", spawn=True)

    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    par = cv2.aruco.DetectorParameters()
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(dic, par)

    fk, lowest_z = make_fk2()
    cam = Realsense(color_res=(1280, 720))
    robot = SOFollower(SOFollowerRobotConfig(port=args.port, id="so101", cameras={}))
    robot.connect()

    samples = []                   # (tid, ang, corners, K)
    seen_at = {}                   # (pan, wf, roll) -> n tags at the base opening
    n_cap = 0

    try:
        cur = read_angles(robot)

        def capture(pan, wf, roll, grip):
            nonlocal cur, n_cap
            tgt = dict(BASE_POSE, shoulder_pan=pan, wrist_flex=wf,
                       wrist_roll=float(roll), gripper=grip)
            tag = f"pan={pan:+.0f} wf={wf:+.0f} roll={roll:+.0f} g={grip:.0f}"
            if lowest_z(tgt) < PLATE_Z + MIN_CLEAR:
                print(f"  {tag}: SKIP (clearance)")
                return 0
            cur = move_to(robot, cur, tgt, seconds=0.8)
            time.sleep(0.45)
            color, depth, K = cam.grab()
            got = detect_tags(det, color)
            n_cap += 1
            rr.set_time_sequence("capture", n_cap)
            vis = color.copy()
            for tid, c in got:
                cv2.polylines(vis, [c.astype(int)], True, (0, 255, 0), 2)
                cv2.putText(vis, f"tag{tid}", tuple(c[0].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, tag, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            rr.log("camera", rr.Image(vis[::2, ::2, ::-1]))
            if not got:
                print(f"  {tag}: no tag")
                return 0
            ang = read_angles(robot)
            for tid, corners in got:
                samples.append((tid, ang, corners, K))
                t_ct = pnp_candidates(corners, K)[0][1]
                rr.log(f"tags/tag{tid}", rr.Points3D([t_ct], radii=0.004))
                print(f"  {tag}: tag{tid} t_cam={np.round(t_ct * 1000).astype(int)}")
            return len(got)

        print("phase 1: roll sweep (fixed opening)")
        for pan, wf, rolls in ROLL_SWEEPS:
            for roll in rolls:
                seen_at[(pan, wf, float(roll))] = capture(pan, wf, float(roll),
                                                          BASE_POSE["gripper"])

        print("phase 2: gripper sweep — best pose per distinct (pan, wf)")
        by_axis = {}               # (pan, wf) -> best (count, roll)
        for (pan, wf, roll), n in seen_at.items():
            if n > 0 and (by_axis.get((pan, wf), (0, 0))[0] < n):
                by_axis[(pan, wf)] = (n, roll)
        best = sorted(by_axis.items(), key=lambda kv: -kv[1][0])[:N_GRIP_POSES]
        if not best:
            print("  no pose detected any tag — nothing to sweep")
        for (pan, wf), (_, roll) in best:
            for grip in GRIPPER_LEVELS:
                capture(pan, wf, roll, grip)
    except Exception as e:
        print(f"\nUnexpected error: {e!r}\nLanding the arm safely...")
    finally:
        cam.stop()
        graceful_shutdown(robot)
        try:
            robot.disconnect()
        except Exception:
            pass

    # persist raw corners so PnP + solver can be re-run offline
    out = ROOT / "outputs/calib/tag_sweep_samples.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump([dict(tid=tid, ang={k: float(v) for k, v in ang.items()},
                    corners=np.asarray(c).tolist(), K=K)
               for tid, ang, c, K in samples], open(out, "w"))
    print(f"samples -> {out}")

    solve(samples)


def solve(samples, write=True):
    """Bundle-adjustment-style joint fit per jaw-assignment hypothesis: DELTA +
    jaw-map scale A + per tag a 6-DoF mount and a print-size scale. Residuals are
    PIXEL reprojection errors of the 4 corners — no per-frame pose extraction, so
    IPPE's two-fold ambiguity and its tilt-dependent range bias (measured up to
    ~80 mm at high tilt on these stickers) never enter the fit; PnP only seeds
    the mounts. Writes outputs/calib/tag_calib.json and a residual figure."""
    import json
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    if len(samples) < 8:
        sys.exit(f"only {len(samples)} detections — not enough.")
    fk, _ = make_fk2()
    R_cb0, t_cb0 = load_handeye()              # seed only — extrinsics are co-fitted
                                               # (solved at ~0.45 m; the sweep runs at
                                               # 0.18-0.31 m, where 1deg ~ 16 px)

    tids = sorted({s[0] for s in samples})
    g_span = np.ptp([s[1]["gripper"] for s in samples])
    print(f"\n{len(samples)} samples across tags {tids}, gripper span {g_span:.0f}")
    if g_span < 5:
        print("WARNING: gripper was not swept — DELTA is gauge-degenerate "
              "with the tag mounts; the fit below is meaningless for DELTA.")

    UNIT = np.array([[-0.5, 0.5, 0], [0.5, 0.5, 0],
                     [0.5, -0.5, 0], [-0.5, -0.5, 0]])

    def unpack(x):
        delta, a = np.degrees(x[0]), x[1]
        R_cb = Rotation.from_rotvec(x[2:5]).as_matrix()
        t_cb = x[5:8]
        tfs = {tid: (Rotation.from_rotvec(x[8 + 7 * k:11 + 7 * k]).as_matrix(),
                     x[11 + 7 * k:14 + 7 * k], x[14 + 7 * k])
               for k, tid in enumerate(tids)}
        return delta, a, (R_cb, t_cb), tfs  # tfs[tid] = (R_mount, t_mount, print_scale)

    def fit(jaw_tid):
        body = {tid: (JAW_BODY if tid == jaw_tid else GRIP_BODY) for tid in tids}

        def resid(x):
            delta, a, (R_cb, t_cb), tfs = unpack(x)
            R_bc, t_bc = R_cb.T, -R_cb.T @ t_cb
            outv = []
            for tid, ang, corners, K in samples:
                R_w, t_w = fk(ang, delta, a)[body[tid]]
                Rm, tm, s = tfs[tid]
                R_ct = R_bc @ R_w @ Rm
                t_ct = R_bc @ (R_w @ tm + t_w) + t_bc
                pts = (R_ct @ (s * TAG_SIDE * UNIT).T).T + t_ct
                uv = pts[:, :2] / pts[:, 2:3]
                uv = uv * (K["fx"], K["fy"]) + (K["ppx"], K["ppy"])
                outv.append((uv - corners).ravel())
            return np.concatenate(outv)

        ext0 = np.concatenate([Rotation.from_matrix(R_cb0).as_rotvec(), t_cb0])
        # seeds: roll offset x jaw-map sign x per-tag PnP candidate for mount init
        first = {tid: next(s for s in samples if s[0] == tid) for tid in tids}
        pnps = {tid: pnp_candidates(first[tid][2], first[tid][3]) for tid in tids}
        best = None
        for d0, a0 in itertools.product((-90.0, 0.0, 90.0, 180.0), (A0, -A0)):
            for choice in itertools.product(*(range(len(pnps[t])) for t in tids)):
                x = [np.radians(d0), a0, *ext0]
                for tid, ci in zip(tids, choice):
                    R_ct0, t_ct0, _ = pnps[tid][ci]
                    R_w0, t_w0 = fk(first[tid][1], d0, a0)[body[tid]]
                    x.extend(Rotation.from_matrix(
                        R_w0.T @ R_cb0 @ R_ct0).as_rotvec())
                    x.extend(R_w0.T @ (R_cb0 @ t_ct0 + t_cb0 - t_w0))
                    x.append(1.0)                  # print scale = true side / TAG_SIDE
                sol = least_squares(resid, np.array(x), method="trf",
                                    loss="soft_l1", f_scale=2.0, max_nfev=6000)
                if best is None or sol.cost < best.cost:
                    best = sol
        return best

    results = {}
    for jaw_tid in tids:
        sol = fit(jaw_tid)
        delta, a, ext, tfs = unpack(sol.x)
        px_rms = float(np.sqrt(np.mean(sol.fun ** 2)))
        results[jaw_tid] = dict(sol=sol, delta=delta, a=a, ext=ext, tfs=tfs,
                                px_rms=px_rms)
        sides = " ".join(f"tag{tid}:{TAG_SIDE * s * 1000:.1f}mm"
                         for tid, (_, _, s) in tfs.items())
        print(f"  hypothesis tag{jaw_tid}=moving jaw: delta={delta:+7.2f}deg "
              f"A={a:+.3f}  reproj {px_rms:.2f}px rms  fitted sides {sides}")

    jaw_tid = min(results, key=lambda k: results[k]["sol"].cost)
    r = results[jaw_tid]
    delta_w = (r["delta"] + 180) % 360 - 180
    R_cb, t_cb = r["ext"]
    dr = np.degrees(np.linalg.norm(Rotation.from_matrix(R_cb0.T @ R_cb).as_rotvec()))
    dt = np.linalg.norm(np.asarray(t_cb) - t_cb0) * 1000
    # px -> mm at the working distance, for intuition (f ~900px, range ~0.25m)
    mm_per_px = 0.25 / 900 * 1000
    print(f"\nbest: tag{jaw_tid} rides the moving jaw")
    print(f"WRIST_ROLL OFFSET = {delta_w:+.2f} deg   jaw map A = {r['a']:+.3f} (B0 {B0:+.0f})")
    print(f"residuals: {r['px_rms']:.2f} px rms (~{r['px_rms'] * mm_per_px:.1f} mm "
          f"at the tag) — the end-to-end FK + detection budget")
    print(f"co-fitted extrinsics moved {dr:.2f} deg / {dt:.1f} mm vs handeye.json")

    if write:                      # selftest must not clobber the real calib
        out = ROOT / "outputs/calib/tag_calib.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(delta_deg=delta_w, jaw_scale_a=r["a"], jaw_b0=B0, jaw_tid=jaw_tid,
                       px_rms=r["px_rms"],
                       R_cam2base=np.asarray(R_cb).tolist(),
                       t_cam2base=np.asarray(t_cb).tolist(),
                       tags={str(tid): dict(body=(JAW_BODY if tid == jaw_tid else GRIP_BODY),
                                            R=R.tolist(), t=np.asarray(t).tolist(),
                                            side_m=TAG_SIDE * float(s))
                             for tid, (R, t, s) in r["tfs"].items()}),
                  open(out, "w"), indent=1)
        print(f"calib -> {out}")

    try:                           # residual figure: per-sample px error vs roll
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        err = r["sol"].fun.reshape(-1, 8)
        px = np.sqrt(np.mean(err ** 2, axis=1))
        rolls = [s[1]["wrist_roll"] for s in samples]
        grips = [s[1]["gripper"] for s in samples]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        col = ["tab:blue" if s[0] == 1 else "tab:orange" for s in samples]
        ax[0].bar(range(len(px)), px, color=col)
        ax[0].set_xlabel("sample (blue=tag1, orange=tag2)")
        ax[0].set_ylabel("reproj err [px]")
        sc = ax[1].scatter(rolls, px, c=grips, cmap="viridis", s=20)
        ax[1].set_xlabel("wrist_roll [deg]"); ax[1].set_ylabel("reproj err [px]")
        plt.colorbar(sc, ax=ax[1], label="gripper")
        ax[0].set_title(f"tag_sweep reprojection residuals — delta={delta_w:+.2f}deg, "
                        f"jaw=tag{jaw_tid}, A={r['a']:+.3f}")
        fig.tight_layout()
        fig.savefig(ROOT / "outputs/calib/tag_sweep_residuals.png", dpi=110)
        print(f"figure -> {ROOT / 'outputs/calib/tag_sweep_residuals.png'}")
    except Exception as e:
        print(f"(residual figure skipped: {e!r})")
    return dict(jaw_tid=jaw_tid, delta=delta_w, a=r["a"], px_rms=r["px_rms"])


def selftest():
    """Synthetic sweep with known DELTA / jaw scale / mounts. Tag poses are
    PROJECTED to pixel corners, and the TRUE camera extrinsics are perturbed
    1.5deg/10mm away from the handeye.json seed — the solver must recover DELTA
    anyway (proves pan variation breaks the camera-vs-DELTA gauge) and identify
    the jaw tag."""
    import cv2
    from scipy.spatial.transform import Rotation
    fk, _ = make_fk2()
    R_cb, t_cb = load_handeye()
    R_cb = Rotation.from_rotvec(np.radians(1.5) * np.array([0.6, -0.64, 0.48])).as_matrix() @ R_cb
    t_cb = t_cb + np.array([0.006, -0.005, 0.006])
    R_bc, t_bc = R_cb.T, -R_cb.T @ t_cb        # base -> cam
    rng = np.random.default_rng(0)
    TRUE_DELTA, TRUE_A, TRUE_JAW = 13.5, 1.27, 2
    # mounts estimated from the real 2026-06-12 sweep samples — realistic facing,
    # so both tags survive the visibility filter like they do on hardware
    mounts = {1: (Rotation.from_rotvec([-2.051, 0.073, -2.039]).as_matrix(),
                  np.array([0.041, 0.12, -0.0677])),
              2: (Rotation.from_rotvec([1.973, -0.117, 1.846]).as_matrix(),
                  np.array([0.006, -0.0566, -0.0297]))}
    K = dict(fx=908.0, fy=908.0, ppx=640.0, ppy=360.0)
    Km = np.array([[K["fx"], 0, K["ppx"]], [0, K["fy"], K["ppy"]], [0, 0, 1]])
    S = TAG_SIDE
    obj = np.array([[-S / 2, S / 2, 0], [S / 2, S / 2, 0],
                    [S / 2, -S / 2, 0], [-S / 2, -S / 2, 0]])

    poses = [(pan, wf, float(r), BASE_POSE["gripper"])
             for pan, wf, rolls in ROLL_SWEEPS for r in rolls]
    poses += [(pan, wf, -20.0, g) for pan, wf in
              ((0.0, -40.0), (0.0, -60.0), (-25.0, -40.0), (25.0, -40.0))
              for g in GRIPPER_LEVELS]
    samples = []
    for pan, wf, roll, grip in poses:
        ang = dict(BASE_POSE, shoulder_pan=pan, wrist_flex=wf,
                   wrist_roll=roll, gripper=grip)
        P = fk(ang, TRUE_DELTA, TRUE_A)
        for tid, (Rm, tm) in mounts.items():
            R_w, t_w = P[JAW_BODY if tid == TRUE_JAW else GRIP_BODY]
            R_bt, t_bt = R_w @ Rm, R_w @ tm + t_w
            R_ct, t_ct = R_bc @ R_bt, R_bc @ t_bt + t_bc
            if (R_ct @ [0, 0, 1])[2] > -0.4:    # tilted >66deg -> realistically undetectable
                continue
            pts = (R_ct @ obj.T).T + t_ct
            uv = (Km @ pts.T).T
            uv = uv[:, :2] / uv[:, 2:3] + rng.normal(0, 0.3, (4, 2))
            samples.append((tid, ang, uv, K))

    res = solve(samples, write=False)
    ok = (res["jaw_tid"] == TRUE_JAW and abs(res["delta"] - TRUE_DELTA) < 0.5
          and abs(res["a"] - TRUE_A) < 0.05)
    print(f"\nselftest: true delta {TRUE_DELTA:+.2f} A {TRUE_A:.3f} jaw=tag{TRUE_JAW} "
          f"-> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
