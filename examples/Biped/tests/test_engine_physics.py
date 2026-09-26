"""
The C++ physics engine against textbook physics: friction on slopes, static friction, rotated-body
inertia, boxes settling flat, and the engine's quaternion convention.
    python tests/test_engine_physics.py
"""
import math

from _common import check, finish, make_world

from bereshit import GameObject, Vector3, BoxCollider, Rigidbody, FixedJoint
import robot as R

print("test_engine_physics")


def box(position, size, mass=1.0, mu=0.6, rotation=(0, 0, 0), **kw):
    return GameObject(position=Vector3(*position), rotation=Vector3(*rotation), size=Vector3(*size)).add_component(
        BoxCollider(), Rigidbody(mass=mass, friction_coefficient=mu, restitution=0.0, **kw))


def floor(mu=0.6):
    return GameObject(position=Vector3(0, -0.5, 0), size=Vector3(40, 1, 40)).add_component(
        BoxCollider(), Rigidbody(isKinematic=True, friction_coefficient=mu, restitution=0.0))


# 1. sliding down a slope: a = g (sin - mu cos), or 0 when friction holds
for deg, mu in ((20, 0.3), (20, 0.6), (35, 0.3), (35, 0.6), (45, 0.6)):
    t = math.radians(deg)
    b = box((0, 0.05, 0), (0.3, 0.1, 0.3), mu=mu)
    world = make_world([b, floor(mu)], gravity=(9.8 * math.sin(t), -9.8 * math.cos(t), 0))
    for _ in range(40):
        world.update(False)
    v0 = b.Rigidbody.velocity.x
    for _ in range(100):
        world.update(False)
    a = (b.Rigidbody.velocity.x - v0) / 0.5
    expected = max(0.0, 9.8 * (math.sin(t) - mu * math.cos(t)))
    check(f"slope {deg} deg, mu {mu}: acceleration matches theory", abs(a - expected) < 0.1,
          f"{a:.2f} vs {expected:.2f} m/s^2")

# 2. static friction holds on a gentle slope (no creep)
t = math.radians(10)
b = box((0, 0.05, 0), (0.2, 0.1, 0.2))
world = make_world([b, floor()], gravity=(9.8 * math.sin(t), -9.8 * math.cos(t), 0))
for _ in range(400):
    world.update(False)
check("no creep on a 10 deg slope with mu 0.6", abs(b.transform.position.x) < 0.001,
      f"moved {b.transform.position.x * 1000:.2f} mm in 2 s")

# 3. a rotated long bar spins about its own long axis when pushed about it, as fast as an unrotated one.
#    Not only at 45 deg: there the world inertia turned the wrong way (R Iinv R^T instead of R^T Iinv R) still
#    gave the right direction; at 30 deg the bar spun 57 deg off its axis at half the speed.
def spin(rotation):
    bar = box((0, 5, 0), (0.02, 0.3, 0.02), rotation=rotation, useGravity=False)
    make_world([bar, box((0, -50, 0), (1, 1, 1), isKinematic=True)], gravity=(0, 0, 0))
    axis = R.local_to_world(bar.transform.quaternion, Vector3(0, 1, 0)).normalized()
    bar.Rigidbody.apply_angular_impulse(axis * 0.001)
    w = bar.Rigidbody.angular_velocity
    return math.degrees(math.acos(min(1.0, abs(w.normalized().dot(axis))))), w.magnitude()


_, upright_speed = spin((0, 0, 0))
for rotation in ((0, 0, 45), (0, 0, 30), (20, 35, 10)):
    angle, speed = spin(rotation)
    check(f"bar rotated {rotation}: spins about the pushed principal axis, at the upright bar's speed",
          angle < 1.0 and abs(speed / upright_speed - 1) < 0.01,
          f"{angle:.2f} deg off, {speed / upright_speed:.3f} of the speed")

# 4. a tilted cube dropped on the floor settles flat on a face
for rot in ((30, 0, 0), (0, 0, 30), (25, 40, 15)):
    cube = box((0, 0.3, 0), (0.1, 0.1, 0.1), rotation=rot)
    world = make_world([cube, floor()])
    for _ in range(800):
        world.update(False)
    q = cube.transform.quaternion
    align = max(abs(R.local_to_world(q, a).y) for a in (Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)))
    check(f"cube tilted {rot} settles flat", align > 0.999 and abs(cube.transform.position.y - 0.05) < 0.003,
          f"alignment {align:.4f}, height {cube.transform.position.y:.4f}")

# 5. quaternion convention: q.conjugate().rotate(v) maps body-local to world (robot.local_to_world)
a = GameObject(position=Vector3(0, 0, 0), size=Vector3(.2, .2, .2)).add_component(
    BoxCollider(), Rigidbody(mass=1, angular_velocity=Vector3(0, 0, 1)))
b = GameObject(position=Vector3(1, 0, 0), size=Vector3(.2, .2, .2)).add_component(
    BoxCollider(), Rigidbody(mass=0.001), FixedJoint(a))
world = make_world([a, b], gravity=(0, 0, 0), epochs=50)
for _ in range(100):
    world.update(True)
offset = (b.transform.position - a.transform.position).normalized()
mapped = R.local_to_world(a.transform.quaternion, Vector3(1, 0, 0))
check("local_to_world matches where a fixed-jointed body really is", (offset - mapped).magnitude() < 0.01,
      f"offset ({offset.x:.3f},{offset.y:.3f}) mapped ({mapped.x:.3f},{mapped.y:.3f})")


# 6. mirror images stay mirror images. Contacts were solved one after another: when a face landed with 4
#    corners, the first-solved corner took the whole impact, and which came first was fixed in world
#    coordinates - a tall box sliding straight forward or back tipped onto one corner and spun clockwise
#    (55 deg in 0.3 s), and the same motion on the left and right foot came out different.
def slide(velocity, rotation=(0, 0, 0)):
    b = box((0, 0.251, 0), (0.1, 0.5, 0.1), rotation=rotation)
    world = make_world([b, floor()], tick=1 / 200, epochs=30)
    b.Rigidbody.velocity = Vector3(*velocity)
    for _ in range(60):
        world.update(False)
    f = R.local_to_world(b.transform.quaternion, Vector3(1, 0, 0))
    p, up = b.transform.position, R.local_to_world(b.transform.quaternion, Vector3(0, 1, 0))
    return math.degrees(math.atan2(-f.z, f.x)), (p.x, p.z, up.x, up.z)


spins = [slide(v)[0] for v in ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1))]
check("a tall box sliding straight doesn't start spinning (< 1 deg in 0.3 s)", max(map(abs, spins)) < 1.0,
      ", ".join(f"{s:+.1f}" for s in spins) + " deg")
_, left = slide((0, 0, 0), rotation=(3, 0, 0))    # tipping over sideways, one way and the other
_, right = slide((0, 0, 0), rotation=(-3, 0, 0))
gap = max(abs(left[0] - right[0]), abs(left[1] + right[1]), abs(left[2] - right[2]), abs(left[3] + right[3]))
check("tipping left == mirror of tipping right", gap < 1e-4, f"largest gap {gap:.1e}")


# 7. the heading in the world doesn't matter: the same motion turned about the vertical comes out the same,
#    turned. It didn't (the world inertia turned the wrong way, rotation steps through Euler angles, friction
#    capped per world axis): the robot turns ~7 deg while lifting the first foot, and the second lift - the
#    same motion at the new heading - fell. Open loop a 7 deg turn moved the pelvis 5.8 mm, 45 deg 42 mm.
def yaw_matrix(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def mat_vec(m, v):
    return [sum(m[i][k] * v[k] for k in range(3)) for i in range(3)]


def as_list(v):
    return [v.x, v.y, v.z]


def orientation(q):
    """Columns: the body's own axes in the world."""
    cols = [as_list(R.local_to_world(q, Vector3(*e))) for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    return [[cols[j][i] for j in range(3)] for i in range(3)]


def set_orientation(body, m):
    """The quaternion whose local_to_world is m (the engine stores the inverse rotation)."""
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = 2 * math.sqrt(1 + t)
        w, x, y, z = s / 4, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s
    else:
        i = max(range(3), key=lambda k: m[k][k])
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * math.sqrt(1 + m[i][i] - m[j][j] - m[k][k])
        v = [0.0, 0.0, 0.0]
        v[i], v[j], v[k] = s / 4, (m[j][i] + m[i][j]) / s, (m[k][i] + m[i][k]) / s
        w, (x, y, z) = (m[k][j] - m[j][k]) / s, v
    q = body.transform.quaternion
    q.w, q.x, q.y, q.z = w, -x, -y, -z
    body.transform.quaternion = q
    body.Cache.SetDirty()                            # the engine caches the rotation matrix and bounding box
    body.Rigidbody.inertia = body.Rigidbody.inertia  # and the world-frame inertia


def turn(bodies, deg, about):
    """Turn bodies rigidly about the vertical through `about`, velocities too."""
    Y = yaw_matrix(deg)
    for b in bodies:
        p = mat_vec(Y, [a - c for a, c in zip(as_list(b.transform.position), about)])
        b.transform.position = Vector3(p[0] + about[0], p[1] + about[1], p[2] + about[2])
        m = orientation(b.transform.quaternion)
        set_orientation(b, [[sum(Y[i][k] * m[k][j] for k in range(3)) for j in range(3)] for i in range(3)])
        b.Rigidbody.velocity = Vector3(*mat_vec(Y, as_list(b.Rigidbody.velocity)))
        b.Rigidbody.angular_velocity = Vector3(*mat_vec(Y, as_list(b.Rigidbody.angular_velocity)))


def box_path(deg, mode):
    """A tilted box spinning in the air, or dropped spinning on a floor without / with friction, turned deg."""
    mu = 0.0 if mode == "no friction" else 0.6
    b = box((0, 0.3 if mode == "in the air" else 0.1, 0), (0.12, 0.03, 0.08), mass=0.3, mu=mu, rotation=(8, 0, 5),
            useGravity=mode != "in the air")
    ground = box((0, -50, 0), (1, 1, 1), isKinematic=True) if mode == "in the air" else floor(mu)
    world = make_world([b, ground], gravity=(0, 0, 0) if mode == "in the air" else (0, -9.8, 0))
    b.Rigidbody.velocity = Vector3(0.3, 0, 0.1)
    b.Rigidbody.angular_velocity = Vector3(3.0, 1.0, 2.0)
    turn([b], deg, [0, 0, 0])
    back = yaw_matrix(-deg)
    path = []
    for _ in range(100):
        world.update(False)
        path.append((mat_vec(back, as_list(b.transform.position)),
                     [[sum(back[i][k] * row[k] for k in range(3)) for row in zip(*orientation(b.transform.quaternion))]
                      for i in range(3)]))
    return path


for mode in ("in the air", "no friction", "friction"):
    base = box_path(0, mode)
    gaps = []
    for deg in (7, 45, 90):
        path = box_path(deg, mode)
        gaps.append(max(max(abs(a - b) for a, b in zip(p0, p1)) for (p0, _), (p1, _) in zip(base, path)))
        gaps.append(0.01 * max(abs(a - b) for (_, m0), (_, m1) in zip(base, path)
                               for r0, r1 in zip(m0, m1) for a, b in zip(r0, r1)))   # 0.01 of the axes ~ 1 cm
    check(f"a box {mode}: turned 7/45/90 deg it moves the same (within 0.1 mm)", max(gaps) < 1e-4,
          f"largest gap {max(gaps) * 1000:.3f} mm")


def robot_path(deg):
    """The robot playing the one-leg reference open loop (the weight over one foot, the other lifting), turned deg."""
    import motions as M
    import WalkAgent as W
    root, rb = R.build_robot()
    world = make_world([root, R.build_floor()], tick=W.TICK, epochs=W.PHYSICS_EPOCHS)
    bodies = [rb["pelvis"]] + list(rb["left"].values()) + list(rb["right"].values())
    world.update(False)
    start = as_list(rb["pelvis"].transform.position)
    turn(bodies, deg, start)
    back = yaw_matrix(-deg)
    path = []
    for step in range(int((M.ONE_LEG_PAUSE + M.ONE_LEG_LEAN + 0.5 * M.ONE_LEG_LIFT) / W.TICK)):
        if step % 4 == 0:
            for s, a in zip(rb["servos"], M.reference_angles("one_leg", step * W.TICK, 0.0, 0.0, first_leg=0)):
                s.command(a)
        world.update(False)
        path.append(mat_vec(back, [a - c for a, c in zip(as_list(rb["pelvis"].transform.position), start)]))
    return path


base = robot_path(0)
gaps = {deg: max(abs(a - b) for p0, p1 in zip(base, robot_path(deg)) for a, b in zip(p0, p1)) for deg in (7, 45, 90)}
check("the robot turned 7/45/90 deg: the same pelvis path through the weight shift and lift (within 0.1 mm)",
      max(gaps.values()) < 1e-4, ", ".join(f"{d} deg {g * 1000:.3f} mm" for d, g in gaps.items()))

finish()
