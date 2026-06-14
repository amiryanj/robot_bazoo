"""Live Realsense viewer in Rerun — max-res color (1280x800) + depth (1280x720).

python vision/rs_view.py            # Ctrl-C to stop
"""
import numpy as np
import pyrealsense2 as rs
import rerun as rr

rr.init("rs_view", spawn=True)

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 800, rs.format.rgb8, 30)
# 848x480, not max: D455 stereo min-Z at 1280x720 is ~50cm — our scene (44-52cm)
# falls inside the blind zone (measured 2026-06-13, vision/depth_check.py)
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
profile = pipe.start(cfg)
scale = profile.get_device().first_depth_sensor().get_depth_scale()
print(f"streaming: color 1280x800 + depth 1280x720 @30  (depth scale {scale} m/unit)")

try:
    while True:
        frames = pipe.wait_for_frames()
        c, d = frames.get_color_frame(), frames.get_depth_frame()
        if not c or not d:
            continue
        rr.set_time_seconds("time", frames.get_timestamp() / 1000.0)
        rr.log("color", rr.Image(np.asanyarray(c.get_data())))
        rr.log("depth", rr.DepthImage(np.asanyarray(d.get_data()), meter=1.0 / scale))
except KeyboardInterrupt:
    pass
finally:
    pipe.stop()
