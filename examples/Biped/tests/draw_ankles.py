"""
Close-up drawing of the simulated robot's feet, ankles and shins, as built (robot.build_robot), without a
render window: a front view and a top view, both legs, joint axes marked. Writes ankles.png next to Train.py.
    python tests/draw_ankles.py
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from PIL import Image, ImageDraw

import robot as R

root, rb = R.build_robot()
PX = 2200                     # pixels per meter
PAD = 40
Z_RANGE = (-0.24, 0.24)       # sideways, m (+z = the robot's left)
Y_TOP = 0.30                  # front view: floor up to mid-shin
X_RANGE = (-0.08, 0.23)       # top view: heel to toe
COLORS = {"shin": (120, 160, 220), "ankle_block": (230, 150, 60), "foot": (110, 190, 110)}
PARTS = ("foot", "ankle_block", "shin")

width = int((Z_RANGE[1] - Z_RANGE[0]) * PX) + 2 * PAD
front_h = int(Y_TOP * PX)
top_h = int((X_RANGE[1] - X_RANGE[0]) * PX)
img = Image.new("RGB", (width, front_h + top_h + 4 * PAD), "white")
d = ImageDraw.Draw(img)


def u(z):
    """Front view seen from in front of the robot: its left (+z) is on the picture's right."""
    return PAD + (z - Z_RANGE[0]) * PX


def front(y):
    return PAD + (Y_TOP - y) * PX


top0 = front_h + 3 * PAD


def top(x):
    return top0 + (X_RANGE[1] - x) * PX   # toes up


d.text((PAD, 10), "FRONT VIEW (looking at the robot from the front)", fill="black")
d.line([(PAD, front(0)), (width - PAD, front(0))], fill="gray", width=2)
d.text((PAD, top0 - 25), "TOP VIEW (from above, toes up)", fill="black")
for side in ("left", "right"):
    for name in PARTS:
        body = rb[side][name]
        p, s = body.transform.position, body.transform.size
        c = COLORS[name]
        y0, y1 = p.y - s.y / 2, min(p.y + s.y / 2, Y_TOP)
        d.rectangle([u(p.z - s.z / 2), front(y1), u(p.z + s.z / 2), front(y0)], outline="black", fill=c)
        d.rectangle([u(p.z - s.z / 2), top(p.x + s.x / 2), u(p.z + s.z / 2), top(p.x - s.x / 2)],
                    outline="black", fill=c)
    foot = rb[side]["foot"].transform
    d.text((u(foot.position.z) - 40, front(0) + 8), f"{side.upper()} FOOT", fill="black")
    d.text((u(foot.position.z) - 40, top(X_RANGE[0]) - 15), f"{side.upper()} FOOT", fill="black")
# joint axes (servo anchors): ankle roll / ankle pitch, red dots with a vertical line through the ankle
for servo in rb["servos"]:
    a = servo.anchor
    if a.y < Y_TOP:
        d.ellipse([u(a.z) - 5, front(a.y) - 5, u(a.z) + 5, front(a.y) + 5], fill="red")
        d.ellipse([u(a.z) - 5, top(a.x) - 5, u(a.z) + 5, top(a.x) + 5], fill="red")
for side in ("left", "right"):
    z = rb[side]["foot"].transform.position.z
    foot = rb[side]["foot"].transform
    d.line([(u(z), front(0)), (u(z), front(Y_TOP))], fill="red", width=1)
    d.text((u(z) + 8, front(0.10)), f"ankle axis z={z * 100:+.1f} cm\nfoot center z={foot.position.z * 100:+.1f} cm\n"
           f"foot {foot.size.z * 100:.0f} cm wide", fill="black")
d.text((PAD, top0 + top_h + 10), "red dots = ankle roll / pitch axes (and the other joints above)", fill="black")
out = os.path.join(HERE, "ankles.png")
img.save(out)
print(out)
