#!/usr/bin/env python
"""The 3-face tag box on the hand-held joystick — step 1: WHICH tags are on it.

A 32 mm cube corner with one tag per face (28 mm black square, 2 mm white margin).
Unlike the two loose tags this replaces, the geometry is known a priori: three mutually
perpendicular faces, each centre one half-edge out along its own normal. So the body
model does not need bundle adjustment (`tag_body.py`) -- only the FACE ASSIGNMENT and
each tag's in-plane rotation, which is what the later steps pin down.

This file does none of that yet. It answers the first question only: what dictionary and
what ids are actually on the box, and how well each face reads.

    python vision/box_tags.py debug --source 9        # live window: aim the camera, see why
    python vision/box_tags.py ident --source 9        # scan dictionaries, then live report
"""
import argparse
import json
import itertools
import math
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

from tag_pose import make_detector, detect, open_source            # noqa: E402
from tag_body import candidates                                   # noqa: E402

MIN_HITS = 10                      # below this a "detection" is background texture
IDS = (7, 11, 15)                  # the three faces (ident, 2026-08-31); 17/37 were 1-frame
DICT = "DICT_4X4_50"               # phantoms. Whitelisting them keeps a phantom out of the
EDGE = 0.032                       # cube edge, m                              arm command.
SIDE = 0.028                       # printed BLACK square per face, m
SAMPLES = ROOT / "outputs/calib/box_samples.json"
MODEL = ROOT / "outputs/calib/box_body.json"

# One per family: 4x4_50 ids also decode inside 4x4_100/250/1000, so listing those would
# just report the same tag several times.
CANDIDATE_DICTS = ["DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_50", "DICT_7X7_50",
                   "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_16H5", "DICT_APRILTAG_25H9",
                   "DICT_APRILTAG_36H10", "DICT_APRILTAG_36H11"]


def tag_px(corners):
    """Apparent black-square side in pixels — mean of the four edges."""
    c = np.asarray(corners)
    return float(np.mean(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)))


def scan_dicts(src, hit_frames=15, patience=60.0):
    """Try every candidate dictionary until `hit_frames` frames actually contained a tag.

    Counting plain frames instead would end the scan in ~1 s -- before the box is even in
    front of the camera. So the budget is spent on frames that carry information, and the
    loop just waits (up to `patience`) for the rest."""
    dets = {d: make_detector(d) for d in CANDIDATE_DICTS}
    hits = {d: Counter() for d in CANDIDATE_DICTS}
    got = seen = 0
    t0 = last = time.perf_counter()
    while seen < hit_frames and time.perf_counter() - t0 < patience:
        frame, _ = src.grab()
        if frame is None:
            continue
        got += 1
        any_hit = False
        for d, det in dets.items():
            for tid, _ in detect(det, frame):
                hits[d][tid] += 1
                any_hit = True
        seen += any_hit
        if time.perf_counter() - last >= 0.5:
            last = time.perf_counter()
            print(f"\r  {time.perf_counter() - t0:4.1f}s  {got} frames, "
                  f"{seen}/{hit_frames} with a tag   ", end="", flush=True)
    print()
    return {d: h for d, h in hits.items() if h}, got


def debug(args):
    """Live Rerun view for aiming and for answering "why did nothing decode?".

    Draws accepted markers green and REJECTED quad candidates red. That distinction is
    the whole diagnosis: red quads on the box means the squares were found but the code
    could not be read (too few pixels, blur, wrong dictionary); no quads at all means the
    box is out of frame, out of focus, or too low-contrast.

    Rerun, not cv2.imshow: this env has opencv-python-HEADLESS (pinned in CLAUDE.md), which
    is built without highgui. cv2's drawing functions still work -- only windowing is gone."""
    import cv2
    import rerun as rr

    src = open_source(args.source, args.fov)
    dict_name = args.dict or "DICT_4X4_50"
    det = make_detector(dict_name)
    rr.init("box_tags_debug", spawn=True)
    print(f"debug view on {dict_name} - Ctrl-C to stop. "
          f"green = decoded, red = quad found but NOT decoded.")

    t0, n, t_fps = time.perf_counter(), 0, time.perf_counter()
    try:
        while True:
            frame, _ = src.grab()
            if frame is None:
                continue
            t = time.perf_counter()
            rr.set_time("time", duration=t - t0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = det.detectMarkers(gray)

            vis = frame.copy()
            cv2.aruco.drawDetectedMarkers(vis, rejected, borderColor=(0, 0, 255))
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids, borderColor=(0, 255, 0))
            rr.log("image", rr.Image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                             .compress(jpeg_quality=75))

            got = sorted(int(i) for i in ids.ravel()) if ids is not None else []
            px = [tag_px(c[0]) for c in corners] if ids is not None else []
            rej_px = [tag_px(c[0]) for c in rejected] if len(rejected) else []
            rr.log("n_decoded", rr.Scalars(len(got)))
            rr.log("n_rejected", rr.Scalars(len(rejected)))
            rr.log("tag_px", rr.Scalars(max(px) if px else 0.0))
            rr.log("biggest_rejected_px", rr.Scalars(max(rej_px) if rej_px else 0.0))
            rr.log("brightness", rr.Scalars(float(gray.mean())))
            rr.log("focus", rr.Scalars(float(cv2.Laplacian(gray, cv2.CV_64F).var())))

            n += 1
            if t - t_fps >= 0.5:
                shown = str(got) if got else "-"
                print(f"\r{n / (t - t_fps):5.1f} Hz  ids {shown:12s} "
                      f"rejected {len(rejected):2d} "
                      f"(biggest {max(rej_px) if rej_px else 0:5.1f} px)  "
                      f"decoded px {[round(v) for v in px] if px else '-'}   ",
                      end="", flush=True)
                n, t_fps = 0, t
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        src.stop()


def _annotate(det, frame):
    """Detect and draw. cv2's DRAWING works in the headless build; only windows do not."""
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = det.detectMarkers(gray)
    vis = frame.copy()
    cv2.aruco.drawDetectedMarkers(vis, rejected, borderColor=(0, 0, 255))
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids, borderColor=(0, 255, 0))
    got = sorted(int(i) for i in ids.ravel()) if ids is not None else []
    px = [tag_px(c[0]) for c in corners] if ids is not None else []
    return vis, got, px, rejected, gray


def debug_window(args):
    """Same detection, shown in a plain SDL window instead of Rerun.

    This exists to answer one question: is the lag in the DETECTION or in the VIEWER?
    pygame, not cv2.imshow -- this env has opencv-python-HEADLESS, which is built without
    highgui, so imshow does not exist. pygame is already a dependency (the gamepad).
    The overlay shows the age of the frame being drawn, which is the actual lag."""
    import os
    import cv2
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    import pygame

    src = open_source(args.source, args.fov)
    det = make_detector(args.dict or DICT)
    frame, _ = src.grab()
    h, w = frame.shape[:2]
    scale = min(1.0, 1280 / w)
    size = (int(w * scale), int(h * scale))
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("box_tags debug (pygame)")
    font = pygame.font.SysFont("monospace", 18)
    print(f"pygame window {size[0]}x{size[1]} - close it or press q to stop")

    n, t_fps, hz = 0, time.perf_counter(), 0.0
    try:
        while True:
            t_grab = time.perf_counter()
            frame, _ = src.grab()
            if frame is None:
                continue
            ms_grab = (time.perf_counter() - t_grab) * 1e3
            t_d = time.perf_counter()
            vis, got, px, rejected, _ = _annotate(det, frame)
            ms_det = (time.perf_counter() - t_d) * 1e3

            t_s = time.perf_counter()
            if scale != 1.0:
                vis = cv2.resize(vis, size)
            surf = pygame.image.frombuffer(
                cv2.cvtColor(vis, cv2.COLOR_BGR2RGB).tobytes(), size, "RGB")
            screen.blit(surf, (0, 0))
            lines = [f"{hz:5.1f} Hz   ids {got if got else '-'}",
                     f"grab {ms_grab:5.1f} ms   detect {ms_det:5.1f} ms   "
                     f"show {(time.perf_counter()-t_s)*1e3:4.1f} ms",
                     f"tag px {[round(v) for v in px] if px else '-'}   "
                     f"rejected {len(rejected)}"]
            for i, ln in enumerate(lines):
                screen.blit(font.render(ln, True, (0, 0, 0)), (13, 11 + 22 * i))
                screen.blit(font.render(ln, True, (255, 255, 0)), (12, 10 + 22 * i))
            pygame.display.flip()

            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN
                                             and e.key in (pygame.K_q, pygame.K_ESCAPE)):
                    raise KeyboardInterrupt
            n += 1
            t = time.perf_counter()
            if t - t_fps >= 0.5:
                hz, n, t_fps = n / (t - t_fps), 0, t
    except KeyboardInterrupt:
        print("stopped")
    finally:
        src.stop()
        pygame.quit()


def ident(args):
    src = open_source(args.source, args.fov)
    for _ in range(10):
        src.grab()                                   # let exposure settle

    if args.dict:
        chosen = args.dict
        print(f"dictionary forced to {chosen}")
    else:
        print(f"Hold the box in view. Scanning {len(CANDIDATE_DICTS)} dictionaries "
              f"(waits for you -- up to {args.patience:.0f} s).")
        hits, n = scan_dicts(src, patience=args.patience)
        for d, h in sorted(hits.items(), key=lambda kv: -sum(kv[1].values())):
            tot = sum(h.values())
            note = "  <- too few, treating as phantom" if tot < MIN_HITS else ""
            print(f"  {d:22s} ids {dict(sorted(h.items()))}  ({tot} hits / {n} frames){note}")
        # A real tag held in view reads in nearly every frame; ARUCO_ORIGINAL in particular
        # fires on background texture a few times per hundred frames. Without this floor an
        # empty scene "identifies" a dictionary off 3 phantoms (observed 2026-08-31).
        real = {d: h for d, h in hits.items() if sum(h.values()) >= MIN_HITS}
        if not real:
            src.stop()
            raise SystemExit(f"no tag read reliably in {args.patience:.0f} s - hold the box "
                             f"in view, closer, and check lighting/focus")
        chosen = max(real, key=lambda d: sum(real[d].values()))
        print(f"-> using {chosen}")

    det = make_detector(chosen)
    seen = Counter()
    px = defaultdict(list)
    pairs = Counter()
    n_visible = Counter()
    frames = 0
    t0 = last = time.perf_counter()
    print(f"\nRotate the box slowly through all three faces for {args.seconds:.0f} s "
          f"(Ctrl-C to stop early).")
    try:
        while (t := time.perf_counter() - t0) < args.seconds:
            frame, _ = src.grab()
            if frame is None:
                continue
            frames += 1
            tags = detect(det, frame)
            ids = sorted({tid for tid, _ in tags})
            for tid, c in tags:
                seen[tid] += 1
                px[tid].append(tag_px(c))
            n_visible[len(ids)] += 1
            for combo in combinations(ids, 2):
                pairs[combo] += 1
            if time.perf_counter() - last >= 0.5:
                last = time.perf_counter()
                print(f"\r{t:5.1f}s  frames {frames:4d}  now: {ids if ids else '-'}"
                      f"   seen: {dict(sorted(seen.items()))}      ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()

    print(f"\n\n--- {frames} frames, dictionary {chosen} ---")
    if not seen:
        print("no tags detected")
        return
    for tid in sorted(seen):
        p = np.array(px[tid])
        print(f"  id {tid:4d}: {seen[tid]:5d} frames ({100*seen[tid]/frames:5.1f}%)  "
              f"black square {p.mean():5.1f} px  (min {p.min():.0f}, max {p.max():.0f})")
    print("\n  tags visible per frame: "
          + ", ".join(f"{k}: {100*v/frames:.1f}%" for k, v in sorted(n_visible.items())))
    print("  co-visible pairs (these are what tie the faces together):")
    if pairs:
        for combo, c in pairs.most_common():
            print(f"    {combo}: {c:5d} frames ({100*c/frames:5.1f}%)")
    else:
        print("    NONE - no two faces were ever visible at once")


# -- step 2: measure the real relative geometry of the three faces ---------------------

def capture(args):
    """Record tag corners over many viewpoints -> SAMPLES (json), for offline fitting."""
    src = open_source(args.source, args.fov)
    det = make_detector(DICT)
    keep = set(IDS)
    for _ in range(10):
        src.grab()

    samples, t0, last = [], time.perf_counter(), 0.0
    print(f"Rotate the box slowly for {args.seconds:.0f} s. What matters is time spent on "
          f"the EDGES, where two or three faces are visible at once -- single-face frames "
          f"say nothing about how the faces relate.")
    try:
        while (t := time.perf_counter() - t0) < args.seconds:
            frame, K = src.grab()
            if frame is None:
                continue
            tags = detect(det, frame, keep)
            if len(tags) >= 2:                       # only co-visible frames carry geometry
                samples.append({"t": round(t, 3), "K": K,
                                "tags": {str(tid): np.asarray(c).tolist() for tid, c in tags}})
            if t - last >= 0.5:
                last = t
                per = Counter(int(i) for s_ in samples for i in s_["tags"])
                print(f"\r{t:5.1f}s  co-visible frames {len(samples):4d}   "
                      f"per tag {dict(sorted(per.items()))}     ", end="", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        src.stop()

    SAMPLES.parent.mkdir(parents=True, exist_ok=True)
    SAMPLES.write_text(json.dumps(samples))
    pairs = Counter(tuple(sorted(int(i) for i in s_["tags"])) for s_ in samples)
    print(f"\nsaved {len(samples)} co-visible frames -> {SAMPLES}")
    for combo, c in sorted(pairs.items()):
        print(f"  {combo}: {c}")


def quat_of(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()


def clusters(Rs, ts, k=2, thresh_deg=8.0):
    """The k largest rotation clusters, greedily peeled off -> [(R, t, n_inliers)].

    Each co-visible frame yields FOUR relative poses (two IPPE candidates per tag). The
    true pairing repeats across viewpoints; so, it turns out, does the both-flipped one.
    Measured 2026-08-31: the top two clusters come in at 25.0% each (286 vs 285, 265 vs
    265, 518 vs 518) -- statistically INDISTINGUISHABLE by vote count. So a per-pair
    argmax picks arbitrarily, and picking independently per pair produced a body whose
    loop closed at 178 deg, i.e. one pair silently flipped. Hence: keep both, let loop
    closure choose (`resolve`)."""
    from scipy.spatial.transform import Rotation
    q = np.array([Rotation.from_matrix(R).as_quat() for R in Rs])
    ts = np.asarray(ts)
    live = np.ones(len(q), bool)
    out = []
    for _ in range(k):
        idx = np.where(live)[0]
        if len(idx) < 10:
            break
        A = np.degrees(2 * np.arccos(np.clip(np.abs(q[idx] @ q[idx].T), 0.0, 1.0)))
        inl = A < thresh_deg
        i = int(inl.sum(axis=1).argmax())
        m = idx[inl[i]]
        qs = q[m] * np.sign(q[m] @ q[idx[i]])[:, None]      # hemisphere-align, then mean
        _, v = np.linalg.eigh(qs.T @ qs)
        out.append((Rotation.from_quat(v[:, -1]).as_matrix(), np.median(ts[m], axis=0),
                    int(len(m))))
        live[m] = False
    return out


def pair_candidates(samples, a, b, k=2):
    """Top-k relative-pose clusters for tag b in tag a's frame."""
    Rs, ts = [], []
    for s_ in samples:
        ca, cb = s_["tags"].get(str(a)), s_["tags"].get(str(b))
        if ca is None or cb is None:
            continue
        for Ra, ta, _ in candidates(ca, s_["K"], SIDE):
            for Rb, tb, _ in candidates(cb, s_["K"], SIDE):
                Rs.append(Ra.T @ Rb)
                ts.append(Ra.T @ (tb - ta))
    return (clusters(Rs, ts, k), len(Rs)) if Rs else ([], 0)


def ang_deg(R):
    return math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def resolve(samples, ids):
    """Pick, per pair, the cluster that makes the three transforms a RIGID BODY.

    A rigid body must satisfy T(a->b) * T(b->c) == T(a->c) exactly. That constraint is
    independent of anything we assume about the box, and it separates the true poses from
    their flips cleanly (measured: 0.25 mm vs 3.72 mm translation closure) where vote
    count cannot."""
    a, b, c = ids
    C = {}
    for pair in ((a, b), (b, c), (a, c)):
        C[pair], n = pair_candidates(samples, *pair)
        if not C[pair]:
            raise SystemExit(f"tags {pair} never co-visible - cannot tie the box together")
    best = None
    for i, j, k in itertools.product(*(range(len(C[p])) for p in ((a, b), (b, c), (a, c)))):
        R1, t1, n1 = C[(a, b)][i]
        R2, t2, n2 = C[(b, c)][j]
        R3, t3, n3 = C[(a, c)][k]
        dt = float(np.linalg.norm(t1 + R1 @ t2 - t3))
        if best is None or dt < best[0]:
            best = (dt, ang_deg(R1 @ R2 @ R3.T), (R1, t1, n1), (R2, t2, n2), (R3, t3, n3))
            pick = (i, j, k)
    return best, C, pick


def fit(args):
    """Measure the box, check it against the ideal cube, and write the body model."""
    samples = json.loads(SAMPLES.read_text())
    a, b, c = IDS
    (dt, dR, P_ab, P_bc, P_ac), C, pick = resolve(samples, IDS)
    print(f"{len(samples)} co-visible frames from {SAMPLES}\n")
    print(f"rigid-body check (the flip is resolved by this, not by vote count):")
    print(f"  loop {a}->{b}->{c} vs {a}->{c}:  rotation {dR:.2f} deg, "
          f"translation {dt*1e3:.2f} mm")
    runner = min((float(np.linalg.norm(C[(a,b)][i][1] + C[(a,b)][i][0] @ C[(b,c)][j][1]
                                       - C[(a,c)][k][1]))
                  for i, j, k in itertools.product(*(range(len(C[p]))
                      for p in ((a,b),(b,c),(a,c))))
                  if (i, j, k) != pick), default=None)
    print(f"  chose cluster combination {pick}; next-best closes at "
          f"{runner*1e3:.2f} mm ({runner/dt:.0f}x worse)\n" if runner else "")

    # body frame == reference tag (IDS[0]), matching tag_body's gauge choice
    model = {a: (np.eye(3), np.zeros(3)), b: (P_ab[0], P_ab[1]), c: (P_ac[0], P_ac[1])}
    h = EDGE / 2
    ideal_gap = h * math.sqrt(2)
    print(f"measured, in tag {a}'s frame (ideal cube corner: normals 90 deg apart, "
          f"centres {ideal_gap*1e3:.1f} mm, face centre z = {-h*1e3:.0f} mm):")
    for tid in (b, c):
        R, t = model[tid]
        tilt = math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))
        print(f"  tag {tid:2d}: normal {tilt:6.2f} deg from tag {a} (ideal 90)   "
              f"centre ({t[0]*1e3:6.1f},{t[1]*1e3:6.1f},{t[2]*1e3:6.1f}) mm, "
              f"|t| {np.linalg.norm(t)*1e3:5.2f}")
    Rbc, tbc = P_bc[0], P_bc[1]
    print(f"  tag {b} -> {c}: normal {ang_deg_normals(Rbc):6.2f} deg (ideal 90)   "
          f"|t| {np.linalg.norm(tbc)*1e3:5.2f} mm")

    # Split the residual: a wrong SIDE is a UNIFORM scale error and inflates every
    # component alike, while a box whose side panels sit outside the top panel inflates
    # only the LATERAL offset. Measured 2026-08-31: lateral +7.3%, axial +0.4% -- so the
    # 28 mm tag size is right and the box is ~1.2 mm thicker than an ideal corner. Do NOT
    # "fix" this by rescaling SIDE; the measured model already carries it.
    lat = [math.hypot(model[t][1][0], model[t][1][1]) for t in (b, c)]
    ax = [abs(model[t][1][2]) for t in (b, c)]
    print(f"\nshape of the residual vs an ideal {EDGE*1e3:.0f} mm corner:")
    print(f"  lateral offset {np.mean(lat)*1e3:5.2f} mm ({100*(np.mean(lat)/h-1):+.1f}% "
          f"vs {h*1e3:.0f})   axial offset {np.mean(ax)*1e3:5.2f} mm "
          f"({100*(np.mean(ax)/h-1):+.1f}%)")
    if abs(np.mean(lat)/h - 1) > 3 * abs(np.mean(ax)/h - 1) + 0.01:
        print(f"  -> lateral only: the tag SIDE ({SIDE*1e3:.0f} mm) is right; the faces sit "
              f"{(np.mean(lat)-h)*1e3:.2f} mm proud (panel thickness). Model absorbs it.")
    else:
        print(f"  -> both inflated: this IS a scale error. Measure the black square with "
              f"calipers; SIDE is probably not {SIDE*1e3:.0f} mm.")

    out = {"dict": DICT, "ref_id": a, "ref_side_m": SIDE,
           "tags": {str(t): {"R": model[t][0].tolist(), "t": model[t][1].tolist(),
                             "side_m": SIDE} for t in IDS},
           "n_frames": len(samples), "loop_rot_deg": dR, "loop_trans_mm": dt * 1e3}
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    MODEL.write_text(json.dumps(out, indent=1))
    print(f"\nsaved -> {MODEL}  (schema matches tag_body.load_model)")


def ang_deg_normals(R):
    return math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))


def cube_edges(lo, hi):
    """The 12 edges of an axis-aligned box as line strips."""
    import itertools as it
    v = [np.array([x, y, z]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
         for z in (lo[2], hi[2])]
    return [[a, b] for a, b in it.combinations(v, 2)
            if np.sum(np.abs(a - b) > 1e-12) == 1]


def show(args):
    """Static 3-D of the fitted model: the three tag faces in the body frame, against
    the ideal cube corner. This is the eyeball check that the fit is physical."""
    import rerun as rr
    from tag_body import load_model, unit_corners

    model = load_model(MODEL)
    rr.init("box_model", spawn=True)
    # body frame == tag IDS[0]'s frame: x right, y up, z out of the face
    rr.log("box", rr.ViewCoordinates.RUB, static=True)

    h = EDGE / 2
    colors = {IDS[0]: (230, 60, 60), IDS[1]: (60, 200, 90), IDS[2]: (70, 130, 240)}

    # ideal 32 mm corner: top face centred on the origin, cube hanging in -z
    rr.log("box/ideal_cube",
           rr.LineStrips3D(cube_edges(np.array([-h, -h, -2 * h]), np.array([h, h, 0.0])),
                           colors=[(120, 120, 120)], radii=0.0002))
    rr.log("box/ideal_centres",
           rr.Points3D([[0, 0, 0], [h, 0, -h], [0, h, -h]], radii=0.0012,
                       colors=[(160, 160, 160)],
                       labels=["ideal top", "ideal +x face", "ideal +y face"]))

    for tid, (R, t, side) in model["tags"].items():
        c = unit_corners(side) @ R.T + t
        rr.log(f"box/tag_{tid}/outline",
               rr.LineStrips3D([np.vstack([c, c[:1]])], colors=[colors.get(tid, (255, 255, 255))],
                               radii=0.0004))
        rr.log(f"box/tag_{tid}/frame",
               rr.Transform3D(translation=t, mat3x3=R, axis_length=0.012))
        rr.log(f"box/tag_{tid}/centre",
               rr.Points3D([t], radii=0.0015, colors=[colors.get(tid, (255, 255, 255))],
                           labels=[f"id {tid}"]))

    print(f"model {MODEL}")
    print(f"  body frame = tag {model_ref(model)}; grey = ideal {EDGE*1e3:.0f} mm corner, "
          f"coloured = measured")
    for tid, (R, t, side) in sorted(model["tags"].items()):
        print(f"  tag {tid:2d}: centre ({t[0]*1e3:6.1f},{t[1]*1e3:6.1f},{t[2]*1e3:6.1f}) mm  "
              f"normal ({(R @ [0, 0, 1.0])[0]:+.3f},{(R @ [0, 0, 1.0])[1]:+.3f},"
              f"{(R @ [0, 0, 1.0])[2]:+.3f})  side {side*1e3:.1f} mm")
    print("\nRerun window is open. Ctrl-C to exit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("stopped")


def model_ref(model):
    for tid, (R, t, _) in model["tags"].items():
        if np.allclose(t, 0) and np.allclose(R, np.eye(3)):
            return tid
    return IDS[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="ident",
                    choices=["ident", "debug", "window", "capture", "fit", "show"])
    ap.add_argument("--source", default="9", help="V4L2 index, or 'realsense'")
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--dict", help="skip the dictionary scan and use this one")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--patience", type=float, default=60.0,
                    help="how long the dictionary scan waits for a tag to appear")
    args = ap.parse_args()
    {"ident": ident, "debug": debug, "window": debug_window, "capture": capture,
     "fit": fit, "show": show}[args.mode](args)


if __name__ == "__main__":
    main()
