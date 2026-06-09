"""
ADXL345 (GY-291) cradle for the SO-101 wrist_flex link.

Run headless:
    freecadcmd adxl_wrist_cradle.py
Outputs adxl_wrist_cradle.stl next to this file.

Coordinate convention of the printed part (mm):
    X = along the PCB long edge (header exits -X)
    Y = along the PCB short edge
    Z = up out of the flat mounting back (board sits face-up in the pocket;
        the flat BOTTOM at Z=0 is what bonds/straps to the wrist side wall)
All dims are parameters — measure your board and tweak, then rerun.
"""
import Part, Mesh, MeshPart
from FreeCAD import Vector
import os

# ---- GY-291 board (MEASURE yours; these are typical) ----
PCB_L      = 21.0    # long edge
PCB_W      = 16.0    # short edge
PCB_T      = 1.6     # board thickness captured by the pocket
FIT        = 0.5     # clearance added around PCB in the pocket

# ---- cradle ----
WALL       = 1.8     # pocket wall thickness
FLOOR      = 2.0     # base thickness under the PCB (the bonding face)
POCKET_H   = 4.5     # how far the walls rise above the floor
LIP        = 0.8     # retaining lip overhang at the top of the long walls
HEADER_GAP = True    # leave the -X short wall open for the pin header

# ---- mounting aids ----
ZIP_SLOT_W = 3.5     # zip-tie slot width
ZIP_SLOT_L = 2.2     # zip-tie slot length (along X)
SCREW_D    = 3.2     # optional M3 clearance holes through the base
SCREW_INSET= 4.0     # screw hole inset from the ends

pocket_l = PCB_L + FIT
pocket_w = PCB_W + FIT
outer_l  = pocket_l + 2 * WALL
outer_w  = pocket_w + 2 * WALL
outer_h  = FLOOR + POCKET_H

# solid outer block, centered in X/Y, base at Z=0
block = Part.makeBox(outer_l, outer_w, outer_h,
                     Vector(-outer_l/2, -outer_w/2, 0))

# pocket cut (open top)
pocket = Part.makeBox(pocket_l, pocket_w, POCKET_H + 1,
                      Vector(-pocket_l/2, -pocket_w/2, FLOOR))
part = block.cut(pocket)

# header exit: open the -X short wall down to the floor
if HEADER_GAP:
    notch = Part.makeBox(WALL + 1, pocket_w - 2, POCKET_H + 1,
                         Vector(-outer_l/2 - 0.5, -(pocket_w - 2)/2, FLOOR))
    part = part.cut(notch)

# retaining lips along the two long (+/-Y) walls at the top
def lip(y_sign):
    return Part.makeBox(pocket_l - 4, LIP + WALL, 0.8,
                        Vector(-(pocket_l - 4)/2,
                               y_sign * (pocket_w/2 - 0) - (LIP + WALL) * (y_sign < 0),
                               outer_h - 0.8))
# simpler: a thin inward ledge on each long wall
ledgeY = LIP
for s in (+1, -1):
    y0 = s * (pocket_w/2) - (ledgeY if s > 0 else 0)
    ledge = Part.makeBox(pocket_l - 4, ledgeY, 0.8,
                         Vector(-(pocket_l - 4)/2, y0, outer_h - 0.8))
    part = part.fuse(ledge)

# zip-tie slots through the floor, just inside each long wall
for s in (+1, -1):
    slot = Part.makeBox(ZIP_SLOT_L, ZIP_SLOT_W, FLOOR + 2,
                        Vector(-ZIP_SLOT_L/2,
                               s * (pocket_w/2 - WALL - ZIP_SLOT_W - 0.5),
                               -1))
    part = part.cut(slot)

# optional M3 screw holes through the base, near the two short ends
for sx in (+1, -1):
    hole = Part.makeCylinder(SCREW_D/2, FLOOR + 2,
                             Vector(sx * (outer_l/2 - SCREW_INSET), 0, -1))
    part = part.cut(hole)

part = part.removeSplitter()

out_dir = os.path.dirname(os.path.abspath(__file__))
stl_path = os.path.join(out_dir, "adxl_wrist_cradle.stl")
m = MeshPart.meshFromShape(Shape=part, LinearDeflection=0.1, AngularDeflection=0.5)
m.write(stl_path)
print("wrote", stl_path)
print("outer envelope (mm): %.1f x %.1f x %.1f" % (outer_l, outer_w, outer_h))
