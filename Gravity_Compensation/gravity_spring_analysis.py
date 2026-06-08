"""
SO-101 gravity-torque analysis for spring / gravity-compensation design.

All link masses, centers of mass (COM) and joint geometry are taken directly
from Simulation/SO101/so101_new_calib.urdf. The URDF link masses ALREADY
INCLUDE the STS3215 servo that is physically part of each link, so no weighing
is required for a first pass.

What it does
------------
1. Forward kinematics of the serial chain (URDF fixed-axis RPY convention).
2. Gravity torque about every revolute joint, for any configuration:
       tau_j = z_j . SUM_i ( (p_com_i - o_j) x (m_i * g) )
   over every link i downstream of joint j.
3. Sweeps the two gravity-dominated joints (shoulder_lift, elbow_flex) over
   their full range and reports the worst-case static torque to compensate.

Edit PAYLOAD_KG to add a mass held at the gripper.
"""

import numpy as np

g_vec = np.array([0.0, 0.0, -9.81])   # base z is up (arm bolted to a table)

# Extra mass held at the gripper frame, kg. Set to your typical payload.
PAYLOAD_KG = 0.0

# ----------------------------------------------------------------------------
# Link inertial data: mass [kg] and COM in the link's own frame [m].
# Straight from the URDF <inertial> blocks.
# ----------------------------------------------------------------------------
LINKS = {
    "base_link":        (0.147,    [ 0.0137179, -5.19711e-05,  0.0334843]),
    "shoulder_link":    (0.100006, [-0.0307604, -1.66727e-05, -0.0252713]),
    "upper_arm_link":   (0.103,    [-0.0898471, -0.00838224,   0.0184089]),
    "lower_arm_link":   (0.104,    [-0.0980701,  0.00324376,   0.0182831]),
    "wrist_link":       (0.079,    [-0.000103312,-0.0386143,   0.0281156]),
    "gripper_link":     (0.087,    [ 0.000213627, 0.000245138,-0.025187 ]),
    "moving_jaw":       (0.012,    [-0.00157495, -0.0300244,   0.0192755]),
}

# ----------------------------------------------------------------------------
# Joints: name -> (parent, child, xyz origin, rpy origin).  Axis is +z for all.
# Ordered along the chain. 'gripper' (the jaw) branches off gripper_link.
# ----------------------------------------------------------------------------
JOINTS = [
    ("shoulder_pan",  "base_link",      "shoulder_link",  [ 0.0388353, -8.97657e-09,  0.0624],     [ 3.14159, 4.18253e-17, -3.14159]),
    ("shoulder_lift", "shoulder_link",  "upper_arm_link", [-0.0303992, -0.0182778,   -0.0542],     [-1.5708, -1.5708, 0.0]),
    ("elbow_flex",    "upper_arm_link", "lower_arm_link", [-0.11257,   -0.028,        1.73763e-16],[ 0.0, 0.0, 1.5708]),
    ("wrist_flex",    "lower_arm_link", "wrist_link",     [-0.1349,     0.0052,       3.62355e-17],[ 0.0, 0.0, -1.5708]),
    ("wrist_roll",    "wrist_link",     "gripper_link",   [ 5.55112e-17,-0.0611,      0.0181],     [ 1.5708, 0.0486795, 3.14159]),
    ("gripper",       "gripper_link",   "moving_jaw",     [ 0.0202,     0.0188,      -0.0234],     [ 1.5708, -5.24284e-08, -1.41553e-15]),
]

# Gripper frame (where a payload effectively hangs), relative to gripper_link.
GRIPPER_FRAME_XYZ = [-0.0079, -0.000218121, -0.0981274]

JOINT_LIMITS = {  # radians, from URDF
    "shoulder_pan":  (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex":    (-1.69,    1.69),
    "wrist_flex":    (-1.65806, 1.65806),
    "wrist_roll":    (-2.74385, 2.84121),
    "gripper":       (-0.174533,1.74533),
}

# Which links sit downstream of (i.e. are moved by) each joint.
DOWNSTREAM = {
    "shoulder_pan":  ["shoulder_link","upper_arm_link","lower_arm_link","wrist_link","gripper_link","moving_jaw"],
    "shoulder_lift": ["upper_arm_link","lower_arm_link","wrist_link","gripper_link","moving_jaw"],
    "elbow_flex":    ["lower_arm_link","wrist_link","gripper_link","moving_jaw"],
    "wrist_flex":    ["wrist_link","gripper_link","moving_jaw"],
    "wrist_roll":    ["gripper_link","moving_jaw"],
    "gripper":       ["moving_jaw"],
}


def rpy_to_R(rpy):
    """URDF fixed-axis convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz @ Ry @ Rx


def T(xyz, rpy):
    M = np.eye(4)
    M[:3,:3] = rpy_to_R(rpy)
    M[:3, 3] = xyz
    return M


def Rz_h(q):
    c, s = np.cos(q), np.sin(q)
    M = np.eye(4)
    M[:3,:3] = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return M


def forward_kinematics(q):
    """q: dict joint_name->angle(rad). Returns world 4x4 transform per link
    and per joint origin."""
    link_T = {"base_link": np.eye(4)}
    joint_world = {}
    for name, parent, child, xyz, rpy in JOINTS:
        origin_world = link_T[parent] @ T(xyz, rpy)   # joint frame before rotation
        joint_world[name] = origin_world
        link_T[child] = origin_world @ Rz_h(q.get(name, 0.0))
    return link_T, joint_world


def link_com_world(link_T):
    coms = {}
    for name,(m,c) in LINKS.items():
        Tl = link_T[name]
        coms[name] = (m, (Tl @ np.array([*c,1.0]))[:3])
    return coms


def gravity_torques(q):
    """Static gravity torque [N.m] about each joint axis, signed."""
    link_T, joint_world = forward_kinematics(q)
    coms = link_com_world(link_T)

    # payload as a point mass at the gripper frame
    if PAYLOAD_KG > 0:
        pf = (link_T["gripper_link"] @ np.array([*GRIPPER_FRAME_XYZ,1.0]))[:3]
        coms["_payload"] = (PAYLOAD_KG, pf)

    tau = {}
    for name, parent, child, xyz, rpy in JOINTS:
        Tj = joint_world[name]
        o = Tj[:3,3]
        zaxis = Tj[:3,:3] @ np.array([0,0,1.0])   # joint axis in world
        downstream = list(DOWNSTREAM[name])
        if PAYLOAD_KG > 0 and "moving_jaw" in downstream:
            downstream = downstream + ["_payload"]
        tot = np.zeros(3)
        for ln in downstream:
            m, p = coms[ln]
            tot += np.cross(p - o, m * g_vec)
        tau[name] = float(zaxis @ tot)
    return tau


def downstream_summary(joint, q):
    """Total downstream mass and the horizontal COM lever arm from the joint
    axis (the number you use for a first lever-arm spring sizing)."""
    link_T, joint_world = forward_kinematics(q)
    coms = link_com_world(link_T)
    if PAYLOAD_KG > 0:
        pf = (link_T["gripper_link"] @ np.array([*GRIPPER_FRAME_XYZ,1.0]))[:3]
        coms["_payload"] = (PAYLOAD_KG, pf)
    o = joint_world[joint][:3,3]
    zaxis = joint_world[joint][:3,:3] @ np.array([0,0,1.0])
    downstream = list(DOWNSTREAM[joint])
    if PAYLOAD_KG > 0:
        downstream += ["_payload"]
    M = 0.0; msum = np.zeros(3)
    for ln in downstream:
        m,p = coms[ln]; M += m; msum += m*p
    com = msum / M
    r = com - o
    r_perp = r - (r @ zaxis)*zaxis          # component perpendicular to axis
    return M, np.linalg.norm(r_perp), com


NM_TO_KGCM = 10.197

def sweep(joint, others=None, n=37):
    others = others or {}
    lo, hi = JOINT_LIMITS[joint]
    print(f"\n=== {joint}: gravity torque over range "
          f"[{np.degrees(lo):.0f}, {np.degrees(hi):.0f}] deg ===")
    print(f"(other joints at: "
          f"{ {k:round(np.degrees(v),0) for k,v in others.items()} or 'all 0'})")
    worst = (0.0, None)
    print(f"{'angle[deg]':>10} {'tau[N.m]':>10} {'tau[kg.cm]':>11}")
    for q_deg in np.linspace(np.degrees(lo), np.degrees(hi), n):
        q = dict(others); q[joint] = np.radians(q_deg)
        t = gravity_torques(q)[joint]
        if abs(t) > abs(worst[0]):
            worst = (t, q_deg)
        if int(round(q_deg)) % 15 == 0:
            print(f"{q_deg:10.1f} {t:10.3f} {t*NM_TO_KGCM:11.2f}")
    print(f"  WORST CASE: {worst[0]:+.3f} N.m  ({worst[0]*NM_TO_KGCM:+.2f} kg.cm)"
          f"  at {worst[1]:.1f} deg")
    return worst


if __name__ == "__main__":
    print(f"PAYLOAD_KG = {PAYLOAD_KG}")
    print(f"Total moving mass (no payload) = "
          f"{sum(m for _,(m,_) in [(0,LINKS[k]) for k in LINKS if k!='base_link'])*0+sum(LINKS[k][0] for k in LINKS if k!='base_link'):.3f} kg")

    # Lever-arm summary for the two joints you'd spring-assist first.
    for j in ("shoulder_lift", "elbow_flex"):
        M, r, com = downstream_summary(j, {})
        print(f"\n[{j}] downstream mass M = {M*1000:.0f} g, "
              f"lever arm r = {r*1000:.1f} mm  ->  "
              f"M*g*r = {M*9.81*r:.3f} N.m ({M*9.81*r*NM_TO_KGCM:.2f} kg.cm) worst case")

    # Full sweeps. Worst elbow load happens with the arm extended; check a few.
    sweep("shoulder_lift", others={"elbow_flex": 0.0})
    sweep("elbow_flex",    others={"shoulder_lift": 0.0})
    # Elbow with the forearm stretched out horizontally (heavier load case):
    sweep("elbow_flex",    others={"shoulder_lift": np.radians(90)})
    sweep("wrist_flex",    others={})
