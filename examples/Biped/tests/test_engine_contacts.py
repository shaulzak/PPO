"""
The C++ engine's contact solver and joint anchors in the cases that broke before: a body resting on several
different bodies, a body that starts deep in the floor, and a hinge cast onto a rotated body.
    python tests/test_engine_contacts.py
"""
import math

from _common import check, finish, make_world

from bereshit import GameObject, Vector3, BoxCollider, Rigidbody, HingeJoint

print("test_engine_contacts")


def box(position, size, mass=1.0, rotation=(0, 0, 0), kinematic=False):
    body = Rigidbody(isKinematic=True, friction_coefficient=0.8, restitution=0.0) if kinematic else \
        Rigidbody(mass=mass, friction_coefficient=0.8, restitution=0.0)
    return GameObject(position=Vector3(*position), rotation=Vector3(*rotation), size=Vector3(*size)).add_component(
        BoxCollider(), body)


# 1. a plank across several floor tiles rests on them. Each contact's push was shared only among the contacts
#    with the same tile, so with 3+ tiles the plank got several full pushes per pass: it was launched (3 tiles),
#    fell through (4) or blew up (6).
for tiles in (2, 3, 4, 6):
    for epochs in (1, 5, 30):
        width = 1.0 / tiles
        floor = [box((-0.5 + width * (i + 0.5), -0.5, 0), (width, 1, 1), kinematic=True) for i in range(tiles)]
        plank = box((0, 0.1, 0), (1, 0.2, 0.5))
        world = make_world(floor + [plank], epochs=epochs)
        heights = []
        for _ in range(400):
            world.update(False)
            heights.append(plank.transform.position.y)
        low, high = min(heights[200:]), max(heights[200:])
        check(f"plank on {tiles} tiles, {epochs} epochs: rests at its height",
              abs(low - 0.1) < 0.005 and abs(high - 0.1) < 0.005, f"{low:.4f}..{high:.4f}, rest 0.1")

# 2. a box that starts 5 cm in the floor comes out without jumping. The penetration recovery velocity
#    (0.2 / dt * depth, ~1.9 m/s here) stayed on the body and threw it 15 cm up.
for depth in (0.02, 0.05):
    floor = box((0, -0.5, 0), (10, 1, 10), kinematic=True)
    b = box((0, 0.1 - depth, 0), (0.4, 0.2, 0.4))
    world = make_world([floor, b])
    peak = 0.0
    for _ in range(200):
        world.update(False)
        peak = max(peak, b.transform.position.y - 0.1)
    check(f"box spawned {depth * 100:.0f} cm deep comes out without jumping", peak < 0.005,
          f"rose {peak * 100:.2f} cm above rest")

# 3. a hinge without an explicit anchor is cast onto the other body's surface, on the line between the two
#    centres. The raycast turned the box the wrong way and returned the hit in its local frame, so on a
#    rotated body the pivot came out off that line and the arm swung about the wrong point.
for angle in (30, 60):
    # the ray from the arm at x = 0.5 along -x hits this box (half size 0.1 x 0.05) at x = min(0.1 / cos, 0.05 / sin)
    a = math.radians(angle)
    surface = min(0.1 / math.cos(a), 0.05 / math.sin(a))
    base = box((0, 0, 0), (0.2, 0.1, 0.1), rotation=(0, 0, angle), kinematic=True)
    arm = box((0.5, 0, 0), (0.1, 0.1, 0.1))
    arm.add_component(HingeJoint(base, Vector3(0, 0, 1)))
    world = make_world([base, arm])
    radius = 0.5 - surface
    worst, lowest = 0.0, 0.0
    for _ in range(200):
        world.update(True)
        p = arm.transform.position
        worst = max(worst, abs(math.hypot(p.x - surface, p.y) - radius))
        lowest = min(lowest, p.y)
    check(f"hinge cast onto a box rotated {angle} deg: the arm swings about the surface point",
          worst < 0.002 and lowest < -0.2, f"radius off by up to {worst * 1000:.1f} mm, swung down to {lowest:.2f}")

finish()
