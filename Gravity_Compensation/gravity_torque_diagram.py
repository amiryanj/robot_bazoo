"""Draw a labeled side-view of the SO-101 arm with per-joint gravity torque.
Saves gravity_torque_diagram.png."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gravity_spring_analysis as G

NM2KG = 10.197

def fk_points(q, payload=0.0):
    G.PAYLOAD_KG = payload
    link_T, jw = G.forward_kinematics(q)
    order = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
    pts = [np.zeros(3)] + [jw[n][:3,3] for n in order]
    tip = (link_T["gripper_link"] @ np.array([*G.GRIPPER_FRAME_XYZ,1.0]))[:3]
    pts.append(tip)
    coms = G.link_com_world(link_T)
    return np.array(pts), coms, link_T

# Worst-case static torque over the full workspace, precomputed by the 2D/3D
# sweep in gravity_spring_analysis (kept here so the figure renders instantly).
WORST = {
    0.0: {"shoulder_lift":0.864, "elbow_flex":0.453, "wrist_flex":0.117},
    0.1: {"shoulder_lift":1.266, "elbow_flex":0.741, "wrist_flex":0.215},
}
def worst_case(joint, payload=0.0):
    return WORST[payload][joint]

# Straight horizontal pose = most intuitive (torque ~ weight x horizontal reach)
q = {"shoulder_lift":0.0,"elbow_flex":0.0,"wrist_flex":0.0}
pts, coms, link_T = fk_points(q, payload=0.0)

# project onto the arm's vertical plane: horizontal = reach along arm azimuth, vert = z
horiz_dir = pts[-1][:2]-pts[0][:2]; horiz_dir/=np.linalg.norm(horiz_dir)
def proj(P): return np.array([P[:2]@horiz_dir, P[2]])
P2 = np.array([proj(p) for p in pts])

fig, ax = plt.subplots(figsize=(12,7))
# links
ax.plot(P2[:,0]*1000, P2[:,1]*1000, "-", lw=6, color="#888", zorder=1, solid_capstyle="round")

# link COMs (dot size ~ mass)
mass_g = {"shoulder_link":100,"upper_arm_link":103,"lower_arm_link":104,
          "wrist_link":79,"gripper_link":87,"moving_jaw":12}
for ln,mg in mass_g.items():
    c = proj(coms[ln][1])
    ax.scatter(c[0]*1000,c[1]*1000,s=mg*4,color="#1f77b4",alpha=0.55,zorder=2,edgecolors="k",lw=0.5)
    ax.annotate(f"{mg} g",(c[0]*1000,c[1]*1000),(0,-16),textcoords="offset points",
                ha="center",fontsize=8,color="#1f77b4")

joint_names = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
# precompute worst-case torques (no payload and +100 g)
wc0 = {j:worst_case(j,0.0) for j in ("shoulder_lift","elbow_flex","wrist_flex")}
wc1 = {j:worst_case(j,0.1) for j in ("shoulder_lift","elbow_flex","wrist_flex")}

labels = {
 "shoulder_pan":  ("shoulder_pan\n(vertical axis)","~0  (gravity\nnegligible)","#555"),
 "shoulder_lift": ("shoulder_lift",
    f"M↓=385 g\nτmax={wc0['shoulder_lift']:.2f} N·m "
    f"({wc0['shoulder_lift']*NM2KG:.1f} kg·cm)\n+100g → {wc1['shoulder_lift']:.2f} N·m","#d62728"),
 "elbow_flex":    ("elbow_flex",
    f"M↓=282 g\nτmax={wc0['elbow_flex']:.2f} N·m "
    f"({wc0['elbow_flex']*NM2KG:.1f} kg·cm)\n+100g → {wc1['elbow_flex']:.2f} N·m","#ff7f0e"),
 "wrist_flex":    ("wrist_flex",
    f"M↓=178 g\nτmax={wc0['wrist_flex']:.2f} N·m "
    f"({wc0['wrist_flex']*NM2KG:.1f} kg·cm)","#2ca02c"),
 "wrist_roll":    ("wrist_roll","~0","#555"),
 "gripper":       ("gripper(jaw)","~0","#555"),
}
offsets = {"shoulder_pan":(-72,18),"shoulder_lift":(-145,30),"elbow_flex":(15,-120),
           "wrist_flex":(-10,62),"wrist_roll":(50,-46),"gripper":(42,30)}
for i,jn in enumerate(joint_names):
    p = P2[i+1]*1000
    name,val,col = labels[jn]
    big = jn in ("shoulder_lift","elbow_flex","wrist_flex")
    ax.scatter(*p,s=160 if big else 70,color=col,zorder=4,edgecolors="k",lw=1.2)
    ax.annotate(f"{name}\n{val}",(p[0],p[1]),offsets[jn],textcoords="offset points",
        ha="center",fontsize=9.5 if big else 8,fontweight="bold" if big else "normal",
        color=col,bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=col,alpha=0.9),zorder=5,
        arrowprops=dict(arrowstyle="->",color=col,lw=1.3))

# gravity arrow
ax.annotate("",(60,P2[:,1].min()*1000-30),(60,P2[:,1].min()*1000+30),
            arrowprops=dict(arrowstyle="-|>",color="k",lw=2))
ax.text(72,P2[:,1].min()*1000-30,"g",fontsize=13,fontweight="bold")

ax.set_title("SO-101 follower — static gravity torque per joint\n"
             "shown in the q=0 calibration pose  •  τmax = worst case over the full joint range\n"
             "total moving mass 485 g  •  link masses already include the STS3215 servos  •  M↓ = mass downstream of the joint",
             fontsize=10.5, pad=14)
ax.set_xlabel("horizontal reach from base [mm]"); ax.set_ylabel("height [mm]")
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.margins(0.20)
plt.tight_layout()
plt.savefig("gravity_torque_diagram.png",dpi=130)
print("saved gravity_torque_diagram.png")
