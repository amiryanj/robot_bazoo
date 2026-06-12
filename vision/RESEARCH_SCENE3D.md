# Research memo: 3-D scene recognition for a home arm (survey 2026-06-12)

**Question:** what should our RGB-D scene-understanding stack adopt or learn from, given
the goal — a systematic, shippable scene model for tabletop arms — and the constraints:
8 GB GPU, fixed top-down D455 (+ wrist cam later), `transformers==4.49` pin, the
`scene_model.json` contract, and a strong preference for measured adoption over
framework tourism.

Current state for reference: support planes + detector-free blob proposals (geometry),
student-YOLO/GDINO (semantics), known-radius sphere fit (metric), cross-modal
self-check (blob↔detector agreement within 7 mm).

## Tier 1 — adopt, in this order

### 1. Florence-2 (Microsoft) — name the blobs, this week
~0.23 B (base) / 0.77 B (large) single model: open-vocab detection, region captioning,
grounding. Our blobs are *class-agnostic*; Florence-2's **region captioning gives every
blob card a cheap semantic label** ("orange ball", "cable", "robot arm") without
training anything — the missing semantics half of the geometry-semantics fusion Javad
asked for. Tiny enough to coexist with everything on 8 GB; likely runs on our pinned
transformers (probe below). Risk: license is MIT, quality on tabletop close-ups to be
measured.

### 2. SAM 3 (Meta, weights released 2025-11) — the new teacher
Unified text/exemplar-promptable detection + segmentation + **tracking** in images and
video; reported 2× over prior promptable-segmentation systems; this collapses our
GDINO(+mask) teacher pipeline into one model and adds the exemplar-prompt trick (show
it one crop of our ball → it finds the ball henceforth, no prompt engineering).
Integration note: too new for our pins — evaluate in an **isolated venv** (the
DeepArUco pattern) as the *offline teacher* for `autolabel.py`; the in-loop student
stays our tiny YOLO. Tracking also matters later: auto-labeling **episode videos** for
VLA data.

### 3. Contact-GraspNet (NVIDIA) — when objects stop being spheres
Depth point cloud → dense 6-DoF parallel-jaw grasp proposals, class-agnostic, trained
on 17 M simulated grasps; community **PyTorch ports** avoid the original TF-1.x pain.
Today our grasp is analytic (sphere + known radius — keep it, it's better for the
ball); the moment the task set includes the spool, cups, blocks, this is the standard
open piece. (**AnyGrasp** is stronger + temporally smooth but **machine-locked
license** — note and skip unless we hit Contact-GraspNet's limits.)

## Tier 2 — architectures to learn from (steal ideas, not repos)

### ConceptGraphs — what scene_model.json wants to grow into
Open-vocab **object-centric 3-D scene graph**: class-agnostic masks fused across views
into per-object 3-D segments + VLM captions + spatial-relation edges. Our scene model
is a single-view mini version. Worth borrowing concretely: (a) **per-object CLIP
embedding** in the object card → open-vocab queries ("pick the orange ball" = cosine
match, no retraining); (b) cross-scan object association (same object ⇄ same card over
time); (c) relation edges (`on(ball, spool)` — we already compute resting_plane, one
step further).

### OK-Robot (NYU) — the systems lesson for "ship to more homes"
Zero-shot pick-and-drop in 10 real homes, 58.5%, from *existing* open models (no new
training) — the headline finding is that **integration quality and error compounding
dominate**, not model choice. Their failure breakdown (semantic memory stale > grasp >
navigation) is a checklist for us. Validates our pattern: modular, measured, contracts
between stages.

### DovSG (RA-L 2025) — localized scene updates
Dynamic scene graphs with **local re-scan after the robot changes the scene** — exactly
our situation after every pick. Borrow the idea, not the repo: after a grasp, re-scan
only the workspace box around the action, patch `scene_model.json` instead of full
re-scan.

## Tier 3 — noted, not pursued now
- **SAM 2 / Grounded-SAM-2** — superseded by SAM 3 for our use; FastSAM already
  measured (93% vs 51% fit purity, centre unchanged — masks reserved for non-spherical
  objects).
- **OpenWorldSAM, OGScene3D, HOV-SG, KeySG** — scene-graph variants for mobile robots /
  large scenes; our single fixed camera doesn't need their SLAM machinery.
- **FoundationPose (NVIDIA)** — 6-DoF pose + tracking for *known/captured* objects;
  the right tool when the twin needs true object orientation (spool, toys). Heavy
  (Docker/Isaac ecosystem); revisit when sim-asset capture becomes the bottleneck.
- **DeepArUco++** — trialed 2026-06-12 on our ring-less tags: chance-level decode
  (in-distribution blur/lighting tool, not a structural-damage tool).

## Adoption sequence (concrete)
1. **Probe Florence-2-base on our blob crops** under the pinned transformers (done —
   see below). If it names blobs decently → `scene_model.scan` gains `caption` per
   blob card.
2. SAM 3 in isolated venv → side-by-side vs GDINO on the saved dataset (teacher
   quality benchmark, same protocol as detector_bench).
3. CLIP embedding per object card (openclip ViT-B/32 fits trivially) → open-vocab
   query function over scene_model.
4. Contact-GraspNet PyTorch port → general grasp proposals, gated by the same
   quality-record discipline.

## Sources
[DovSG](https://github.com/BJHYZJ/DovSG) · [ConceptGraphs](https://concept-graphs.github.io/) ·
[OK-Robot](https://ok-robot.github.io/) · [SAM 3](https://ai.meta.com/research/sam3/) ·
[SAM 3 @ ultralytics](https://docs.ultralytics.com/models/sam-3) ·
[Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) ·
[FoundationPose](https://github.com/NVlabs/FoundationPose) ·
[Contact-GraspNet](https://github.com/NVlabs/contact_graspnet) ·
[Contact-GraspNet PyTorch](https://github.com/elchun/contact_graspnet_pytorch) ·
[AnyGrasp SDK (licensed)](https://github.com/graspnet/anygrasp_sdk) ·
[OpenWorldSAM](https://arxiv.org/abs/2507.05427)
