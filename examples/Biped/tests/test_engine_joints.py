"""
The C++ engine's hinge motor, hinge axis, custom inertia, copies and objects added at runtime.
    python tests/test_engine_joints.py
"""
import copy
import math

from _common import check, finish, make_world

from bereshit import GameObject, Vector3, BoxCollider, Rigidbody, HingeJoint, FixedJoint

print("test_engine_joints")

_KEEP = []   # copies aren't owned by any world: keep them alive until the end


def box(position, size, mass=1.0, rotation=(0, 0, 0), kinematic=False, collider=True, rigidbody=True, name=""):
    obj = GameObject(position=Vector3(*position), rotation=Vector3(*rotation), size=Vector3(*size), name=name)
    if collider:
        obj.add_component(BoxCollider())
    if rigidbody:
        obj.add_component(Rigidbody(isKinematic=True, friction_coefficient=0.8, restitution=0.0) if kinematic else
                          Rigidbody(mass=mass, friction_coefficient=0.8, restitution=0.0))
    return obj


def motor_arm(rotation=(0, 0, 0), speed=0.0, torque=5.0):
    """A kinematic base at the origin and an arm at x = 0.5 on a motor hinge about z."""
    base = box((0, 0, 0), (0.2, 0.2, 0.2), kinematic=True, name="base")
    arm = box((0.5, 0, 0), (0.6, 0.1, 0.1), rotation=rotation, name="arm")
    hinge = HingeJoint(base, Vector3(0, 0, 1))
    arm.add_component(hinge)
    hinge.motor_enabled = True
    hinge.motor_speed = speed
    hinge.max_motor_torque = torque
    return base, arm, hinge


def vec(v):
    return (round(v.x, 4), round(v.y, 4), round(v.z, 4))


# 1. a copy of a motor hinge keeps its motor (copy.deepcopy of a robot copies each component with Copy()):
#    the copy had the motor off
base, arm, hinge = motor_arm(speed=1.5, torque=3.0)
copied = copy.copy(hinge)
_KEEP.append(copied)
check("a copied hinge keeps its motor",
      (copied.motor_enabled, copied.motor_speed, copied.max_motor_torque) == (True, 1.5, 3.0),
      f"enabled {copied.motor_enabled}, speed {copied.motor_speed}, max torque {copied.max_motor_torque}")

# 2. an inertia set by hand stays: attach() overwrote one set before add_component, and copies lost it
custom = Vector3(0.3, 0.2, 0.1)
early = Rigidbody(mass=1.0)
early.inertia = custom
GameObject(size=Vector3(1, 1, 1)).add_component(early)
check("an inertia set before add_component is kept", vec(early.inertia) == vec(custom),
      f"{vec(early.inertia)}, set {vec(custom)}")
body = box((0, 0, 0), (1, 1, 1))
body.Rigidbody.inertia = custom
rb_copy = copy.copy(body.Rigidbody)
_KEEP.append(GameObject(size=Vector3(1, 1, 1)).add_component(rb_copy))
check("a copy keeps a body's custom inertia", vec(rb_copy.inertia) == vec(custom),
      f"{vec(rb_copy.inertia)}, set {vec(custom)}")

# 3. the hinge's world axis is the one it turns about. The joint and motor read the axis in the owner's
#    frame, axis_world in the other body's: with the two bodies turned differently they pointed apart.
base, arm, hinge = motor_arm(rotation=(0, 90, 0), speed=2.0)
world = make_world([base, arm], gravity=(0, 0, 0))
for _ in range(50):
    world.update(True)
w = arm.Rigidbody.angular_velocity
axis = hinge.axis_world
along = abs(w.dot(axis)) / max(w.magnitude(), 1e-12)
check("the hinge turns about its axis_world", w.magnitude() > 1.0 and along > 0.99,
      f"spins at {w.magnitude():.2f} rad/s, {math.degrees(math.acos(min(along, 1.0))):.0f} deg off axis_world")

# 4. a negative max motor torque doesn't drive: the clamp to [-max, +max] became a constant +|max|
base, arm, hinge = motor_arm(speed=0.0, torque=-5.0)
world = make_world([base, arm], gravity=(0, 0, 0))
for _ in range(50):
    world.update(True)
w = arm.Rigidbody.angular_velocity.magnitude()
check("a motor held at speed 0 with a negative max torque stays still", w < 1e-6, f"spins at {w:.2f} rad/s")


# 5. an object added to a world at runtime (add_child) behaves like one built into it. World::AddChild never
#    registered its Rigidbody (no gravity, never integrated), solved joints of bodies that aren't in the
#    physics, and took colliders without a Rigidbody, which the contact search then dereferenced.
def falling_scene():
    floor = box((0, -0.5, 0), (10, 1, 10), kinematic=True, name="floor")
    faller = box((0, 1.0, 0), (0.4, 0.4, 0.4), name="faller")
    decor = box((0, 0.1, 0), (0.2, 0.2, 0.2), rigidbody=False, name="decor")   # collider only: not physics
    loose = box((3, 1.0, 0), (0.2, 0.2, 0.2), collider=False, name="loose")     # rigidbody only: not physics
    loose.add_component(HingeJoint(floor, Vector3(0, 0, 1)))
    return floor, [faller, decor, loose]


def run(world, objects):
    for _ in range(200):
        world.update(True)
    return {o.name: vec(o.transform.position) for o in objects}


floor, parts = falling_scene()
built = run(make_world([floor, GameObject(children=parts)]), parts)
floor, parts = falling_scene()
group = GameObject()
world = make_world([floor, group])
for p in parts:
    group.add_child(p)
added = run(world, parts)
check("objects added at runtime move like objects built into the world", added == built,
      f"added {added}, built {built}")


# 6. copy.deepcopy of a robot gives a robot of its own. A copied joint kept the original's other body, so a
#    copy moved away was dragged back to the original robot; a copied FixedJoint came out as a plain Joint that
#    holds nothing; and the C++ deep_copy, which crashed on every Python object, was the only one that remapped.
def robot(joint):
    base = box((0, 0, 0), (0.2, 0.2, 0.2), kinematic=True, name="base")
    part = box((0.5, 0, 0), (0.3, 0.1, 0.1), name="part")
    part.add_component(HingeJoint(base, Vector3(0, 0, 1)) if joint == "hinge" else FixedJoint(base))
    return GameObject(children=[base, part], name="robot")


def moved(obj, dz):
    for o in obj.children:
        o.transform.position = o.transform.position + Vector3(0, 0, dz)
    return obj


def settle(roots):
    world = make_world(roots)
    for _ in range(200):
        world.update(True)


def offset(root):
    base, part = root.children
    return vec(part.transform.position - base.transform.position)


for joint in ("hinge", "fixed"):
    alone = robot(joint)
    settle([alone])
    original = robot(joint)
    clone = moved(copy.deepcopy(original), 5.0)
    settle([original, clone])
    check(f"a copied {joint} joint holds the copy's own body",
          offset(clone) == offset(alone), f"copy's part at {offset(clone)} from its base, {offset(alone)} alone")
    check(f"the original {joint} robot isn't disturbed by its copy",
          offset(original) == offset(alone), f"part at {offset(original)} from its base, {offset(alone)} alone")

# a joint to a body outside the copied objects stays on that body
alone = robot("hinge")
settle([alone])
base = box((0, 0, 0), (0.2, 0.2, 0.2), kinematic=True, name="base")
part = box((0.5, 0, 0), (0.3, 0.1, 0.1), name="part")
part.add_component(HingeJoint(base, Vector3(0, 0, 1)))
part_copy = copy.deepcopy(part)
_KEEP.append(part)
settle([base, part_copy])   # the original part stays out of the world
check("a copied part whose joint's body isn't copied stays on that body",
      vec(part_copy.transform.position - base.transform.position) == offset(alone),
      f"{vec(part_copy.transform.position - base.transform.position)} from the base, {offset(alone)} alone")

check("GameObject has no deep_copy (it crashed on every Python object)", not hasattr(GameObject, "deep_copy"))

finish()
