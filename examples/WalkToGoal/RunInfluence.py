from bereshit.addons.PPO import Config, Academy
from bereshit import GameObject, Vector3, Core, Camera, BoxCollider, Rigidbody, FixedJoint
from bereshit.addons.essentials import FPS_cam, CamController, Servo
from bereshit.addons.PPO.essentials import Goal, Wall
from MoveToGoal import MoveToGoal
from bereshit.addons.PPO.essentials.ServoMovment import ServoMovment

cam = GameObject(position=Vector3(5, 13, -5), rotation=Vector3(45,-30,0)).add_component(Camera(shading="material preview"), CamController(), FPS_cam())

floor = GameObject(position=Vector3(0,-1,0), size=Vector3(150,1,150)).add_component(BoxCollider(), Rigidbody(isKinematic=True, friction_coefficient=1), Wall())

wall1 = GameObject(position=Vector3(-75.5,0,0), size=Vector3(1,1,150)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())
wall2 = GameObject(position=Vector3(75.5,0,0), size=Vector3(1,1,150)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())
wall3 = GameObject(position=Vector3(0,0,-75.5), size=Vector3(150,1,1)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())
wall4 = GameObject(position=Vector3(0,0,75.5), size=Vector3(150,1,1)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())




goal = GameObject(position=Vector3(5,13,5), size=Vector3(1, 1, 1)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Goal())


max = 45
min = -45
speed = 1
torque = 5

feet = GameObject(size=Vector3(2.6, 0.5, 1.2), position=Vector3(-.5, 0, -0.3), name="feet").add_component(
        BoxCollider(), Rigidbody(mass=0.01))

mount = GameObject(position=Vector3(0, 1, 0), size=Vector3(.5, .5, .5), name="mount").add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(feet))

servo1 = GameObject(position=Vector3(0, 2, 0), size=Vector3(.5, .5, .5), name="ankle1").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(mount, Vector3(0,0,1), max, min, speed, torque))

servo2 = GameObject(position=Vector3(0, 3, 0), size=Vector3(.5, .5, .5), name="ankle2").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(servo1,Vector3(1, 0,0), max, min, speed, torque))

calf = GameObject(position=Vector3(0, 5, 0), size=Vector3(.5, 3, .5)).add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(servo2))

mount2 = GameObject(position=Vector3(0, 7, 0), size=Vector3(.5, .5, .5)).add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(calf))

knee = GameObject(position=Vector3(0, 8, 0), size=Vector3(.5, .5, .5), name="knee").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(mount2, Vector3(0, 0,1), max, min, speed, torque))

thigh = GameObject(position=Vector3(0, 10, 0), size=Vector3(.5, 3, .5)).add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(knee))

hip1 = GameObject(position=Vector3(0, 12, 0), size=Vector3(.5, .5, .5), name="hip1").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(thigh,Vector3(1, 0, 0), max, min, speed, torque))  # lower hip servo: sideways (roll)

hip2 = GameObject(position=Vector3(0, 13, 0), size=Vector3(.5, .5, .5), name="hip2").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(hip1,Vector3(0, 0, 1), max, min, speed, torque))  # upper hip servo: forward-back (pitch)

leg1 = GameObject(size=Vector3(), children=[feet, mount, servo1, servo2, calf, mount2, knee, thigh, hip1, hip2])


feet2 = GameObject(size=Vector3(2.6, 0.5, 1.2), position=Vector3(-.5, 0, -0.3), name="feet").add_component(
        BoxCollider(), Rigidbody(mass=0.01))

mount_2 = GameObject(position=Vector3(0, 1, 0), size=Vector3(.5, .5, .5), name="mount").add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(feet2))

servo12 = GameObject(position=Vector3(0, 2, 0), size=Vector3(.5, .5, .5), name="ankle1").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(mount_2, Vector3(0,0,1), max, min, speed, torque))

servo22 = GameObject(position=Vector3(0, 3, 0), size=Vector3(.5, .5, .5), name="ankle2").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(servo12,Vector3(1, 0,0), max, min, speed, torque))

calf2 = GameObject(position=Vector3(0, 5, 0), size=Vector3(.5, 3, .5)).add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(servo22))

mount22 = GameObject(position=Vector3(0, 7, 0), size=Vector3(.5, .5, .5)).add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(calf2))

knee2 = GameObject(position=Vector3(0, 8, 0), size=Vector3(.5, .5, .5), name="knee").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(mount22, Vector3(0, 0,1), max, min, speed, torque))

thigh2 = GameObject(position=Vector3(0, 10, 0), size=Vector3(.5, 3, .5)).add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(knee2))

hip12 = GameObject(position=Vector3(0, 12, 0), size=Vector3(.5, .5, .5), name="hip1").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(thigh2, Vector3(1, 0, 0), max, min, speed, torque))  # lower hip servo: sideways (roll)

hip22 = GameObject(position=Vector3(0, 13, 0), size=Vector3(.5, .5, .5), name="hip2").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(hip12,Vector3(0, 0, 1), max, min, speed, torque))  # upper hip servo: forward-back (pitch)

leg2 = GameObject(size=Vector3(), children=[feet2, mount_2, servo12, servo22, calf2, mount22, knee2, thigh2, hip12, hip22])

feet2 = leg2.search_by_name("feet")[0]

mount2 = leg2.search_by_name("mount")[0]
feet2.transform.position.z += 0.6

mount2.FixedJoint.cast_anchor()

leg2.transform.local_position += Vector3(0, 0, 2)

leg2.set_default()

hip_l = leg2.search_by_name("hip2")[0]

hip_bone = GameObject(position=Vector3(0, 14, 1), size=Vector3(.5, .5, 2), name="hip_bone").add_component(BoxCollider(), Rigidbody(mass=0.05), FixedJoint(hip_l), FixedJoint(hip2))

config = Config(
        obs_dim=286,
        action_dim_continuous=10,
        rollout_steps = 10000,
        device="cpu",
        max_steps=4000,
        best_model_path="model.pt",
)

Academy.setup_trainer(config)
Academy.load_trained_model(model="model.pt")

scene = [floor, wall1, wall2, wall3, wall4, goal, cam]

legs = GameObject(size=Vector3(), children=[leg1, leg2, hip_bone])

control_dt = 1/20  # agent decision interval, must match TrainSingelAgent.py

legs.add_component(MoveToGoal(goal, control_dt))
# legs.add_component(ServoMovment(servo1))


Core.run(scene + [legs], tick=1/240, Render=True, scriptRefreshRate=control_dt)