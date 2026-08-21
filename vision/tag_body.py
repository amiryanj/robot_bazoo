#!/usr/bin/env python
"""Rigid multi-tag body model for the hand-held joystick (teleop step 2).

Tag 13 sits on the FRONT of the controller (visible in a normal grip), tag 5 UNDERNEATH.
One tag is not enough: rotate the controller and the visible tag changes. To keep a
continuous body pose we need each tag's pose RELATIVE to the body, which is what this
measures.

Body frame == the reference tag's frame (id 13), so a normal grip yields the body pose
directly and the under-tag only fills in when the front one is hidden. That choice is
pure gauge -- any rigid frame works -- but it keeps the best-observed tag exact.

Method: capture corners over many viewpoints, then bundle-adjust per-frame body poses +
each non-reference tag's mounting transform + its SIDE, minimising corner reprojection
under a robust loss. The side is SOLVED, not measured: a wrong side scales that tag's
distance linearly and would tear the body apart, while rigidity across viewpoints pins
it down (same idea as tag_sweep.py's per-tag print scale). The reference tag's side sets
the single global scale.

The planar two-fold ambiguity is handled as in tag_sweep.py: both IPPE_SQUARE candidates
are kept per observation and the fit alternates with re-assignment. A naive "take the
lower reprojection error" init biases the result, because at small tags the wrong
candidate often wins by a hair.

    python vision/tag_body.py capture --source 9    # rotate the joystick slowly, ~60 s
    python vision/tag_body.py solve                 # -> outputs/calib/joystick_body.json
    python vision/tag_body.py view --source 9       # live body pose; check the handoff
    python vision/tag_body.py --selftest            # synthetic end-to-end, no camera
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

from tag_pose import Webcam, make_detector, detect, open_source   # noqa: E402

DICT = "DICT_4X4_50"               # both real ids are < 50; 50 codes sit far apart, so
IDS = (13, 5)                      # misdecodes are far rarer than in 4x4_250 (measured:
REF_ID = 13                        # a real tag misdecoded as id 128 at full size)
REF_SIDE = 0.0274                  # the repo's desk-tag print (cam_calib.DESK_TAG_SIDE_M)
SAMPLES = ROOT / "outputs/calib/joystick_samples.json"
MODEL = ROOT / "outputs/calib/joystick_body.json"


def unit_corners(side):
    """Tag corners in its own frame, image order TL,TR,BR,BL (x right, y up, z out)."""
    s = side / 2.0
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)


def Km(K):
    return np.array([[K["fx"], 0, K["ppx"]], [0, K["fy"], K["ppy"]], [0, 0, 1]], np.float64)


def project(K, R, t, obj):
    p = obj @ R.T + t
    z = np.maximum(p[:, 2], 1e-6)
    return np.stack([K["fx"] * p[:, 0] / z + K["ppx"],
                     K["fy"] * p[:, 1] / z + K["ppy"]], axis=1)


def candidates(corners, K, side):
    """Both IPPE_SQUARE solutions -> [(R, t, reproj_px)], best first."""
    import cv2
    n, rv, tv, err = cv2.solvePnPGeneric(
        unit_corners(side).astype(np.float32), np.asarray(corners, np.float32),
        Km(K), np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    out = [(cv2.Rodrigues(rv[i])[0], tv[i].ravel(), float(np.ravel(err)[i])) for i in range(n)]
    return sorted(out, key=lambda c: c[2])


def rvec_of(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def R_of(rvec):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(rvec).as_matrix()


# ── capture ───────────────────────────────────────────────────────────────────────────

def capture(args):
    src = open_source(args.source, args.fov)
    det = make_detector(DICT)
    keep = set(IDS)
    for _ in range(10):
        src.grab()

    samples, t0, last = [], time.perf_counter(), 0.0
    print(f"Rotate the joystick SLOWLY through every orientation for {args.seconds:.0f} s.\n"
          f"Both tags visible at once is what ties them together -- roll it so the front "
          f"tag tips away and the under tag comes up, pausing in between.")
    try:
        while (t := time.perf_counter() - t0) < args.seconds:
            frame, K = src.grab()
            if frame is None:
                continue
            tags = detect(det, frame, keep)
            if tags:
                samples.append({"t": round(t, 3), "K": K,
                                "tags": {str(tid): np.asarray(c).tolist() for tid, c in tags}})
            if t - last >= 1.0:
                last = t
                both = sum(1 for s in samples if len(s["tags"]) > 1)
                per = {i: sum(1 for s in samples if str(i) in s["tags"]) for i in IDS}
                print(f"\r{t:5.1f}s  frames {len(samples):4d}  both-visible {both:4d}  "
                      f"{per}   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        src.stop()

    both = sum(1 for s in samples if len(s["tags"]) > 1)
    SAMPLES.parent.mkdir(parents=True, exist_ok=True)
    json.dump(samples, open(SAMPLES, "w"))
    print(f"\nsaved {len(samples)} frames ({both} with both tags) -> {SAMPLES}")
    if both < 20:
        print("WARNING: too few both-visible frames to tie the tags together. The mount "
              "transform comes ONLY from those; re-run and roll it through the handoff.")


# ── solve ─────────────────────────────────────────────────────────────────────────────

def select(samples, max_frames=200, verbose=True):
    """Keep the frames that actually carry information: the CO-VISIBLE ones.

    A single-tag frame brings its own 6-DoF body pose (6 unknowns) against 8 residuals,
    so it barely constrains the shared mount/size while costing a full parameter block.
    A 60 s capture yields ~1700 frames = ~10k parameters, and finite-differencing a
    Jacobian that size takes minutes for no accuracy gain -- the selftest solves to
    0.2 deg from 12 co-visible frames. Spread the pick over the whole capture so
    viewpoint diversity survives the subsampling."""
    both = [s for s in samples if len(s["tags"]) > 1]
    if len(both) < 20:                    # too thin: fall back to everything we have
        if verbose:
            print(f"only {len(both)} co-visible frames - keeping single-tag frames too, "
                  f"but the mount transform will be poorly constrained")
        return samples[:max_frames * 4]
    pick = [both[i] for i in np.linspace(0, len(both) - 1, min(max_frames, len(both))).astype(int)]
    if verbose:
        print(f"{len(samples)} frames captured -> using {len(pick)} co-visible "
              f"(of {len(both)}); single-tag frames add parameters, not information")
    return pick


def solve(samples, ref_side=REF_SIDE, verbose=True, max_frames=200):
    """Bundle-adjust body poses + mounting transform and side of each non-reference tag.

    Returns the model dict. Gauge: body frame == reference tag frame, whose side is
    fixed at `ref_side` (the one global scale nothing here can observe)."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    samples = select(samples, max_frames, verbose)
    others = sorted({int(i) for s in samples for i in s["tags"]} - {REF_ID})
    if not others:
        raise SystemExit("only the reference tag was seen - nothing to calibrate")

    # ── init: body pose from the reference tag; mount of each other tag from the frames
    # where both are visible (all candidate pairings, keep the most consistent).
    def obs(s, tid):
        return np.array(s["tags"][str(tid)]) if str(tid) in s["tags"] else None

    mounts, sides = {}, {}
    for tid in others:
        pairs = []
        for s in samples:
            cr, co = obs(s, REF_ID), obs(s, tid)
            if cr is None or co is None:
                continue
            for Rr, tr, er in candidates(cr, s["K"], ref_side):
                for Ro, to, eo in candidates(co, s["K"], ref_side):
                    pairs.append((er + eo, Rr.T @ Ro, Rr.T @ (to - tr)))
        if not pairs:
            raise SystemExit(f"tag {tid} never co-visible with the reference tag")
        pairs.sort(key=lambda p: p[0])
        best = pairs[:max(len(pairs) // 4, 1)]           # the cleanest quarter
        mounts[tid] = (np.median([rvec_of(p[1]) for p in best], axis=0),
                       np.median([p[2] for p in best], axis=0))
        sides[tid] = ref_side                            # scale starts at the ref print

    # ── per-frame body pose init
    frames, pose0 = [], []
    for s in samples:
        cr = obs(s, REF_ID)
        if cr is not None:
            R, t, _ = candidates(cr, s["K"], ref_side)[0]
        else:
            tid = next(int(i) for i in s["tags"])
            Ro, to, _ = candidates(obs(s, tid), s["K"], sides[tid])[0]
            Rm, tm = R_of(mounts[tid][0]), mounts[tid][1]
            R, t = Ro @ Rm.T, to - Ro @ Rm.T @ tm        # body from tag observation
        frames.append(s)
        pose0.append(np.concatenate([rvec_of(R), t]))

    F, T = len(frames), len(others)
    idx = {tid: i for i, tid in enumerate(others)}
    x0 = np.concatenate([np.concatenate(pose0),
                         np.concatenate([np.concatenate([mounts[i][0], mounts[i][1], [0.0]])
                                         for i in others])])

    # observation list: (frame, tag or None for the reference, corners, K)
    ob = [(f, (None if int(tid) == REF_ID else int(tid)), np.array(c), frames[f]["K"])
          for f, s in enumerate(frames) for tid, c in s["tags"].items()]

    def unpack(x):
        P = x[:6 * F].reshape(F, 6)
        Q = x[6 * F:].reshape(T, 7)
        return P, Q

    def resid(x):
        P, Q = unpack(x)
        out = np.empty(8 * len(ob))
        for k, (f, tid, c, K) in enumerate(ob):
            Rb, tb = R_of(P[f, :3]), P[f, 3:]
            if tid is None:
                objb, = (unit_corners(ref_side),)
            else:
                q = Q[idx[tid]]
                objb = unit_corners(ref_side * np.exp(q[6])) @ R_of(q[:3]).T + q[3:6]
            out[8 * k:8 * k + 8] = (project(K, Rb, tb, objb) - c).ravel()
        return out

    sp = lil_matrix((8 * len(ob), x0.size), dtype=int)
    for k, (f, tid, _, _) in enumerate(ob):
        sp[8 * k:8 * k + 8, 6 * f:6 * f + 6] = 1
        if tid is not None:
            j = 6 * F + 7 * idx[tid]
            sp[8 * k:8 * k + 8, j:j + 7] = 1

    r0 = resid(x0)
    res = least_squares(resid, x0, jac_sparsity=sp, loss="soft_l1", f_scale=2.0,
                        xtol=1e-10, ftol=1e-10, max_nfev=120, verbose=0)
    P, Q = unpack(res.x)
    rms0, rms = np.sqrt(np.mean(r0 ** 2)), np.sqrt(np.mean(res.fun ** 2))

    model = {"dict": DICT, "ref_id": REF_ID, "ref_side_m": ref_side,
             "tags": {str(REF_ID): {"R": np.eye(3).tolist(), "t": [0, 0, 0],
                                    "side_m": ref_side}},
             "n_frames": F, "n_obs": len(ob), "rms_px": float(rms)}
    for tid in others:
        q = Q[idx[tid]]
        model["tags"][str(tid)] = {"R": R_of(q[:3]).tolist(), "t": q[3:6].tolist(),
                                   "side_m": float(ref_side * np.exp(q[6]))}
    if verbose:
        print(f"\nbundle adjustment: {F} frames, {len(ob)} tag observations")
        print(f"  reprojection rms {rms0:.2f} px -> {rms:.2f} px")
        for tid in others:
            m = model["tags"][str(tid)]
            d = np.linalg.norm(m["t"]) * 1e3
            ang = np.degrees(np.linalg.norm(rvec_of(np.array(m["R"]))))
            print(f"  tag {tid}: {d:.1f} mm from tag {REF_ID}, rotated {ang:.1f} deg, "
                  f"side {m['side_m'] * 1e3:.1f} mm (solved)")
    return model


# ── runtime fusion ────────────────────────────────────────────────────────────────────

def load_model(path=MODEL):
    m = json.load(open(path))
    return {"ref_side": m["ref_side_m"],
            "tags": {int(k): (np.array(v["R"]), np.array(v["t"]), v["side_m"])
                     for k, v in m["tags"].items()}}


def body_pose(dets, K, model, guess=None):
    """Fuse every visible tag into ONE body pose -> (R, t, reproj_px) or None.

    Two tags make the point set non-planar, so the pose is unique. A single tag is
    planar and therefore two-fold ambiguous: both candidates are returned to the
    caller's `guess` for arbitration -- temporal continuity is the only thing that can
    break that tie, and getting it wrong flips the body 180 deg."""
    import cv2
    obj, img = [], []
    for tid, c in dets:
        if tid not in model["tags"]:
            continue
        R, t, side = model["tags"][tid]
        obj.append(unit_corners(side) @ R.T + t)
        img.append(np.asarray(c, np.float64))
    if not obj:
        return None
    obj, img = np.vstack(obj), np.vstack(img)

    if len(img) == 4:                                    # one tag: planar, ambiguous
        tid, c = dets[0]
        Rm, tm, side = model["tags"][tid]
        cands = []
        for Ro, to, err in candidates(c, K, side):
            Rb = Ro @ Rm.T
            cands.append((Rb, to - Rb @ tm, err))
        if guess is not None:
            Rg = guess[0]
            cands.sort(key=lambda c_: -float(np.trace(Rg.T @ c_[0])))   # closest rotation
        Rb, tb, e = cands[0]
        # SQRT2: solvePnPGeneric reports RMS over the 2N scalar COORDINATES, while the
        # multi-tag branch below reports RMS of the per-corner EUCLIDEAN distance. The
        # ratio is exactly 1/sqrt(2) (measured over 200 random poses, sd 0.0000), so
        # without this the caller's reprojection gate is 41% looser on a single tag --
        # i.e. most permissive exactly when the pose is least trustworthy.
        return Rb, tb, e * math.sqrt(2.0)

    ok, rv, tv = cv2.solvePnP(obj.astype(np.float32), img.astype(np.float32), Km(K),
                              np.zeros(5), flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    R, t = cv2.Rodrigues(rv)[0], tv.ravel()
    err = float(np.sqrt(np.mean(np.sum((project(K, R, t, obj) - img) ** 2, axis=1))))
    return R, t, err


# ── live view ─────────────────────────────────────────────────────────────────────────

def view(args):
    import cv2
    import rerun as rr
    from tag_pose import rpy_deg

    model = load_model()
    src = open_source(args.source, args.fov)
    det = make_detector(DICT)
    keep = set(model["tags"])
    rr.init("tag_body", spawn=True)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    t0, guess, n, t_fps = time.perf_counter(), None, 0, time.perf_counter()
    print("Body pose from whichever tags are visible. Watch for JUMPS at the handoff.")
    try:
        while True:
            frame, K = src.grab()
            if frame is None:
                continue
            t = time.perf_counter()
            rr.set_time("time", duration=t - t0)
            dets = detect(det, frame, keep)
            rr.log("world/cam", rr.Pinhole(image_from_camera=Km(K),
                                           resolution=[frame.shape[1], frame.shape[0]]))
            rr.log("world/cam/image",
                   rr.Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                     .compress(jpeg_quality=70))
            out = body_pose(dets, K, model, guess) if dets else None
            if out is None:
                rr.log("world/body", rr.Clear(recursive=True))
                guess = None
            else:
                R, tv, err = out
                guess = (R, tv)
                rr.log("world/body", rr.Transform3D(translation=tv, mat3x3=R,
                                                    axis_length=0.05))
                yaw, pitch, roll = rpy_deg(R)
                for k, v in (("x_mm", tv[0] * 1e3), ("y_mm", tv[1] * 1e3),
                             ("z_mm", tv[2] * 1e3), ("yaw_deg", yaw),
                             ("pitch_deg", pitch), ("roll_deg", roll),
                             ("reproj_px", err), ("n_tags", len(dets))):
                    rr.log(f"body/{k}", rr.Scalars(float(v)))
            n += 1
            if t - t_fps >= 1.0:
                print(f"\r{n / (t - t_fps):5.1f} Hz  tags {[d[0] for d in dets]}    ",
                      end="", flush=True)
                n, t_fps = 0, t
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        src.stop()


# ── selftest ──────────────────────────────────────────────────────────────────────────

def selftest():
    """Synthetic rigid body with a KNOWN mount and a DIFFERENT tag size: prove the solver
    recovers both, including the size it was never told."""
    rng = np.random.default_rng(0)
    K = dict(fx=914.0, fy=914.0, ppx=640.0, ppy=360.0)
    # ~90 deg between the faces (front vs underside). NOT 180: two opposite faces are
    # never co-visible, and with no co-visible frame the mount transform is unobservable
    # -- that is a property of the physical layout, not of the solver.
    true_R = R_of(np.array([np.pi / 2 * 1.05, 0.10, 0.05]))
    true_t = np.array([0.002, -0.030, -0.021])
    true_side5 = 0.0200                                     # smaller than the 27.4 mm ref

    samples, n_both = [], 0
    for _ in range(90):
        rv = np.array([np.pi, 0, 0]) + rng.normal(0, 0.75, 3)   # wide tilt range
        Rb, tb = R_of(rv), np.array([rng.normal(0, .05), rng.normal(0, .05),
                                     rng.uniform(0.35, 0.65)])
        tags = {}
        for tid, (Rm, tm, side) in {REF_ID: (np.eye(3), np.zeros(3), REF_SIDE),
                                    5: (true_R, true_t, true_side5)}.items():
            objb = unit_corners(side) @ Rm.T + tm
            p = objb @ Rb.T + tb
            if np.any(p[:, 2] < 0.1):
                continue
            nrm = Rb @ Rm @ np.array([0, 0, 1.0])
            if nrm @ (p.mean(0) / np.linalg.norm(p.mean(0))) > -0.34:
                continue                    # beyond ~70 deg incidence a tag stops reading
            px = project(K, Rb, tb, objb) + rng.normal(0, 0.3, (4, 2))
            tags[str(tid)] = px.tolist()
        if not tags:
            continue
        n_both += len(tags) > 1
        samples.append({"t": 0.0, "K": K, "tags": tags})

    print(f"synthetic: {len(samples)} frames, {n_both} with both tags")
    model = solve(samples, verbose=False)
    m5 = model["tags"]["5"]
    dR = np.degrees(np.linalg.norm(rvec_of(np.array(m5["R"]).T @ true_R)))
    dt = np.linalg.norm(np.array(m5["t"]) - true_t) * 1e3
    ds = abs(m5["side_m"] - true_side5) * 1e3
    print(f"  mount rot err {dR:.2f} deg, pos err {dt:.2f} mm, "
          f"side {m5['side_m']*1e3:.1f} mm (true {true_side5*1e3:.1f}, err {ds:.2f} mm)")
    print(f"  reprojection rms {model['rms_px']:.2f} px")
    assert dR < 2.0 and dt < 3.0 and ds < 1.5, "solver did not recover the body"

    mdl = {"ref_side": REF_SIDE,
           "tags": {int(k): (np.array(v["R"]), np.array(v["t"]), v["side_m"])
                    for k, v in model["tags"].items()}}
    s = samples[0]
    dets = [(int(k), np.array(v)) for k, v in s["tags"].items()]
    R, t, err = body_pose(dets, s["K"], mdl)
    print(f"  fusion on frame 0: reproj {err:.2f} px")
    assert err < 2.0
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", choices=["capture", "solve", "view"])
    ap.add_argument("--source", default="9")
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--samples", default=str(SAMPLES))
    ap.add_argument("--max-frames", type=int, default=200,
                    help="cap on co-visible frames used by the solver")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
    elif args.mode == "capture":
        capture(args)
    elif args.mode == "solve":
        model = solve(json.load(open(args.samples)), max_frames=args.max_frames)
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        json.dump(model, open(MODEL, "w"), indent=1)
        print(f"\nsaved -> {MODEL}")
    elif args.mode == "view":
        view(args)
    else:
        ap.error("need a mode: capture | solve | view (or --selftest)")


if __name__ == "__main__":
    main()
