#!/usr/bin/env python
"""6-D pose of a hand-held tag board — the teleop input device (step 1: perception only).

Goal: stick AprilTags on a hand-held "joystick", locate it in 6-D from a plain webcam,
and later drive the arm with it. This file is ONLY the perception half — frames in,
tag pose out, Rerun to eyeball it. Nothing robot-related lives here.

Reuses the PnP recipe proven in `vision/tag_sweep.py`: SOLVEPNP_IPPE_SQUARE through
solvePnPGeneric, both planar solutions kept, lower-reprojection one taken. A small tag
genuinely flips between those two poses — that is the planar two-fold ambiguity, not
noise, and ignoring it makes the orientation jump.

Intrinsics: the Realsense reports its own. A plain webcam has none, so K is guessed from
a horizontal FOV (`--fov`). That is fine for the clutched *relative* teleop mapping (it
only rescales how far a hand motion travels); it is NOT fine for absolute geometry.

    python vision/tag_pose.py --selftest            # synthetic end-to-end, no camera
    python vision/tag_pose.py --source 0            # laptop webcam -> Rerun
    python vision/tag_pose.py --source realsense
    python vision/tag_pose.py --print 20 21 22 23   # tag PNGs to print, exact size
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

# 36h11 is the standard AprilTag family: far fewer false positives than 4x4_50 and it
# cannot collide with the arm's existing tags (finger ids 1,2 / desk id 13 are 4x4_50).
DICT = "DICT_APRILTAG_36H11"
SIDE = 0.040                       # printed side of the BLACK square, metres
OUT = ROOT / "outputs/tags"


def make_detector(dict_name=DICT):
    import cv2
    par = cv2.aruco.DetectorParameters()
    # SUBPIX, not CORNER_REFINE_APRILTAG: the apriltag path costs 88 ms/frame at 720p
    # vs 7.7 ms here (measured 2026-08-20), which caps the loop at 6 Hz -- unusable for
    # teleop. Same choice as tag_sweep.py / cam_calib.py.
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    return cv2.aruco.ArucoDetector(dic, par)


def detect(det, bgr, keep=None):
    """-> [(id, corners 4x2)] in image order (TL, TR, BR, BL).

    `keep`: whitelist of ids. Worth using with the big 4x4_250/1000 dictionaries --
    their codes sit closer together and clothing/background texture does produce
    phantom ids (measured on this scene). A phantom tag would be a real arm command."""
    import cv2
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return []
    return [(int(i), c[0].astype(np.float64)) for c, i in zip(corners, ids.ravel())
            if keep is None or int(i) in keep]


def pose(corners, K, side=SIDE):
    """(R_ct, t_ct, reproj_px) — tag pose in the camera frame, best of the two
    IPPE_SQUARE solutions. Tag frame: x right, y up, z out of the tag face."""
    import cv2
    s = side / 2.0
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
    Km = np.array([[K["fx"], 0, K["ppx"]], [0, K["fy"], K["ppy"]], [0, 0, 1]], np.float64)
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj, np.asarray(corners, np.float32), Km, np.zeros(5),
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    errs = np.ravel(errs)
    i = int(np.argmin(errs[:n]))
    return cv2.Rodrigues(rvecs[i])[0], tvecs[i].ravel(), float(errs[i])


def K_from_fov(w, h, fov_deg):
    """Pinhole K guessed from horizontal FOV — square pixels, centred principal point."""
    f = (w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    return dict(fx=f, fy=f, ppx=w / 2.0, ppy=h / 2.0)


def rpy_deg(R):
    """ZYX Euler (yaw, pitch, roll) in degrees — for readable plots only."""
    pitch = np.degrees(np.arcsin(-np.clip(R[2, 0], -1, 1)))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return yaw, pitch, roll


# ── camera sources ────────────────────────────────────────────────────────────────────

class Webcam:
    """Plain V4L2 device, read on a background thread. K is a FOV guess (see docstring).

    The thread is not an optimisation, it is a correctness fix. V4L2 QUEUES frames, so a
    consumer slower than the camera gets progressively STALER ones and the latency grows
    without bound -- measured 2026-08-20: with a 60 ms loop, read() returned in 6 ms
    instead of 33, i.e. handing back frames from the queue rather than the sensor.
    CAP_PROP_BUFFERSIZE=1 does not help; this backend ignores it. So a thread drains the
    camera at full rate and keeps only the newest frame, and grab() blocks until a frame
    NEWER than the last one it handed out exists -- fresh frames, no duplicates.
    """

    def __init__(self, index, size=(1280, 720), fov=70.0):
        import cv2
        self.cap = cv2.VideoCapture(int(index))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open /dev/video{index}")
        # MJPG first: the raw YUYV mode of a UVC cam usually tops out at 640x480, so
        # asking for 720p without it silently gives you a small frame (= a small tag).
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        for _ in range(5):
            self.cap.read()                        # let the format switch settle
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"/dev/video{index} opened but delivers no frames")
        h, w = frame.shape[:2]
        print(f"/dev/video{index}: {w}x{h}, K from a {fov:.0f} deg FOV guess")
        self.K = K_from_fov(w, h, fov)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame, self._seq, self._taken = frame, 0, -1
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self._frame, self._seq = f, self._seq + 1

    def grab(self, timeout=0.5):
        """Newest UNSEEN frame, or (None, K) if none arrived within `timeout`."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            with self._lock:
                if self._seq != self._taken:
                    self._taken = self._seq
                    return self._frame, self.K
            time.sleep(0.001)
        return None, self.K

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


class RS:
    """The D455 (colour only here) — reports true intrinsics per grab."""

    def __init__(self, size=(1280, 720)):
        from handeye_calib import Realsense
        self.cam = Realsense(color_res=size)

    def grab(self):
        color, _depth, K = self.cam.grab()
        return color, K

    def stop(self):
        self.cam.stop()


def open_source(spec, fov):
    """'realsense', 'virtual' (a rendered tag body — no hardware), or a V4L2 index."""
    if spec == "realsense":
        return RS()
    if spec == "virtual":
        from tag_body import VirtualCam          # lazy: tag_body imports this module
        return VirtualCam(fov=fov)
    return Webcam(spec, fov=fov)


# ── printable tags ────────────────────────────────────────────────────────────────────

def print_tags(ids, dict_name, side_m, dpi=300):
    """Write one PNG per id, sized so that printing at `dpi` gives exactly `side_m`
    of black square (plus the mandatory 1-module white quiet zone around it)."""
    import cv2
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    mods = dic.markerSize + 2                      # black square = data + 1 border module
    px_per_mod = max(1, round(side_m * 1000 / 25.4 * dpi / mods))
    px = px_per_mod * mods
    quiet = px_per_mod                             # white quiet zone, 1 module
    OUT.mkdir(parents=True, exist_ok=True)
    for tid in ids:
        img = cv2.aruco.generateImageMarker(dic, int(tid), px)
        page = np.full((px + 2 * quiet, px + 2 * quiet), 255, np.uint8)
        page[quiet:quiet + px, quiet:quiet + px] = img
        path = OUT / f"{dict_name.lower()}_id{tid}_{round(side_m * 1000)}mm.png"
        cv2.imwrite(str(path), page)
        print(f"  {path}")
    print(f"\nPrint at {dpi} DPI with NO scaling ('actual size' / 100%), then MEASURE the "
          f"black square: it must be {side_m * 1000:.0f} mm. Pass the measured value as "
          f"--side (metres) — pose scale is linear in it.")


# ── live view ─────────────────────────────────────────────────────────────────────────

def view(args):
    import cv2
    import rerun as rr

    src = open_source(args.source, args.fov)
    det = make_detector(args.dict)
    keep = set(args.ids) if args.ids else None
    rr.init("tag_pose", spawn=True)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)      # camera frame: x right, y down, z fwd

    t0 = time.perf_counter()
    seen_prev, n, t_fps = set(), 0, time.perf_counter()
    print(f"Watching {args.source} for {args.dict} tags, side {args.side * 1000:.0f} mm"
          f"{', ids ' + str(sorted(keep)) if keep else ''}. Ctrl-C to stop.")
    try:
        while True:
            frame, K = src.grab()
            if frame is None:
                continue
            t = time.perf_counter()
            rr.set_time("time", duration=t - t0)
            tags = detect(det, frame, keep)

            rr.log("world/cam", rr.Pinhole(
                image_from_camera=[[K["fx"], 0, K["ppx"]], [0, K["fy"], K["ppy"]], [0, 0, 1]],
                resolution=[frame.shape[1], frame.shape[0]]))
            rr.log("world/cam/image",
                   rr.Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                     .compress(jpeg_quality=70))

            seen = set()
            for tid, corners in tags:
                R, tvec, err = pose(corners, K, args.side)
                seen.add(tid)
                rr.log(f"world/cam/image/tag_{tid}",
                       rr.LineStrips2D([np.vstack([corners, corners[:1]])], radii=1.5))
                rr.log(f"world/tag_{tid}",
                       rr.Transform3D(translation=tvec, mat3x3=R, axis_length=args.side))
                yaw, pitch, roll = rpy_deg(R)
                for k, v in (("x_mm", tvec[0] * 1e3), ("y_mm", tvec[1] * 1e3),
                             ("z_mm", tvec[2] * 1e3), ("yaw_deg", yaw),
                             ("pitch_deg", pitch), ("roll_deg", roll),
                             ("reproj_px", err)):
                    rr.log(f"tag_{tid}/{k}", rr.Scalars(float(v)))
            for tid in seen_prev - seen:                       # stale poses must not linger
                rr.log(f"world/tag_{tid}", rr.Clear(recursive=True))
                rr.log(f"world/cam/image/tag_{tid}", rr.Clear(recursive=True))
            seen_prev = seen

            n += 1
            if t - t_fps >= 1.0:
                rr.log("rate_hz", rr.Scalars(n / (t - t_fps)))
                print(f"\r{n / (t - t_fps):5.1f} Hz  tags: "
                      f"{sorted(seen) if seen else '-'}      ", end="", flush=True)
                n, t_fps = 0, t
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        src.stop()


# ── selftest ──────────────────────────────────────────────────────────────────────────

def selftest():
    """Render a tag at a known pose, detect it, and check the recovered pose. Proves the
    detect -> PnP path (dictionary, corner order, object-point convention, scale)."""
    import cv2
    w, h = 1280, 720
    K = K_from_fov(w, h, 70.0)
    Km = np.array([[K["fx"], 0, K["ppx"]], [0, K["fy"], K["ppy"]], [0, 0, 1]], np.float64)
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT))
    det = make_detector()

    # A tag facing the camera is NOT R=I: the tag frame is y-up / z-out-of-the-face, the
    # camera frame is y-down / z-forward, so front-facing means a 180 deg flip about x.
    # (Rendering R=I would draw the tag's back, mirrored, and no detector would read it.)
    flip = cv2.Rodrigues(np.array([np.pi, 0.0, 0.0]))[0]
    worst = 0.0
    for i, (tilt, tvec) in enumerate([
            (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.45])),
            (np.array([0.35, -0.25, 0.1]), np.array([0.06, -0.03, 0.60])),
            (np.array([-0.5, 0.4, -0.3]), np.array([-0.09, 0.05, 0.35]))]):
        rvec = cv2.Rodrigues(flip @ cv2.Rodrigues(tilt)[0])[0].ravel()
        s = SIDE / 2
        obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
        px, _ = cv2.projectPoints(obj, rvec, tvec, Km, np.zeros(5))
        px = px.reshape(4, 2)

        marker = cv2.aruco.generateImageMarker(dic, 20, 400)
        q = 400 // (dic.markerSize + 2)                       # quiet zone = 1 module
        card = np.full((400 + 2 * q, 400 + 2 * q), 255, np.uint8)
        card[q:q + 400, q:q + 400] = marker
        # card corners of the BLACK square map to the projected object corners
        src_pts = np.float32([[q, q], [q + 400, q], [q + 400, q + 400], [q, q + 400]])
        H = cv2.getPerspectiveTransform(src_pts, px.astype(np.float32))
        img = cv2.warpPerspective(card, H, (w, h), borderValue=255)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        tags = detect(det, bgr)
        assert len(tags) == 1 and tags[0][0] == 20, f"case {i}: detected {tags and tags[0][0]}"
        R, t, err = pose(tags[0][1], K)
        R_true = cv2.Rodrigues(rvec)[0]
        d_t = np.linalg.norm(t - tvec) * 1e3
        d_R = np.degrees(np.arccos(np.clip((np.trace(R_true.T @ R) - 1) / 2, -1, 1)))
        print(f"  case {i}: pos err {d_t:5.2f} mm, rot err {d_R:5.2f} deg, "
              f"reproj {err:.3f} px")
        assert d_t < 3.0 and d_R < 2.0, f"case {i}: {d_t:.2f} mm / {d_R:.2f} deg"
        worst = max(worst, d_t)
    print(f"selftest OK (worst position error {worst:.2f} mm)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="V4L2 index, or 'realsense'")
    ap.add_argument("--dict", default=DICT)
    ap.add_argument("--side", type=float, default=SIDE, help="tag black-square side, m")
    ap.add_argument("--fov", type=float, default=70.0,
                    help="horizontal FOV guess for webcam intrinsics, degrees")
    ap.add_argument("--print", nargs="+", type=int, metavar="ID",
                    help="write printable PNGs for these tag ids and exit")
    ap.add_argument("--ids", nargs="+", type=int, metavar="ID",
                    help="only accept these tag ids (guards against phantom detections)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
    elif args.print:
        print_tags(args.print, args.dict, args.side)
    else:
        view(args)


if __name__ == "__main__":
    main()
