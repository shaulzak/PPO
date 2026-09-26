from bereshit.addons.PPO import Config, Academy
from bereshit import GameObject, Vector3, Core, Camera, BoxCollider, Rigidbody, FixedJoint
from bereshit.addons.essentials import FPS_cam, CamController, Servo
from bereshit.addons.PPO.essentials import Goal, Wall
from MoveToGoal import MoveToGoal

cam = GameObject(position=Vector3(5, 13, -5), rotation=Vector3(45,-30,0)).add_component(Camera(shading="material preview"), CamController(), FPS_cam())

floor = GameObject(position=Vector3(0,-1,0), size=Vector3(150,1,150)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())

wall1 = GameObject(position=Vector3(-75.5,0,0), size=Vector3(1,1,150)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())
wall2 = GameObject(position=Vector3(75.5,0,0), size=Vector3(1,1,150)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())
wall3 = GameObject(position=Vector3(0,0,-75.5), size=Vector3(150,1,1)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())
wall4 = GameObject(position=Vector3(0,0,75.5), size=Vector3(150,1,1)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())


feet = GameObject(size=Vector3(2.6, 0.5, 1.2), position=Vector3(-.5, 0, -0.3), name="feet").add_component(
        BoxCollider(), Rigidbody(mass=0.01))

goal = GameObject(position=Vector3(5,13,5), size=Vector3(1, 1, 1)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Goal())


max = 45
min = -45
speed = 1
torque = 3

mount = GameObject(position=Vector3(0, 1, 0), size=Vector3(.5, .5, .5), name="mount").add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(feet))

servo1 = GameObject(position=Vector3(0, 2, 0), size=Vector3(.5, .5, .5), name="ankle1").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(mount, Vector3(0,0,1), max, min, speed, torque))

servo2 = GameObject(position=Vector3(0, 3, 0), size=Vector3(.5, .5, .5), name="ankle2").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(servo1,Vector3(1, 0,0), max, min, speed, torque))

calf = GameObject(position=Vector3(0, 5, 0), size=Vector3(.5, 3, .5)).add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(servo2))

mount2 = GameObject(position=Vector3(0, 7, 0), size=Vector3(.5, .5, .5)).add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(calf))

knee = GameObject(position=Vector3(0, 8, 0), size=Vector3(.5, .5, .5), name="knee").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(mount2, Vector3(0, 0,1), max, min, speed, torque))

thigh = GameObject(position=Vector3(0, 10, 0), size=Vector3(.5, 3, .5)).add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(knee))

hip1 = GameObject(position=Vector3(0, 12, 0), size=Vector3(.5, .5, .5), name="hip1").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(thigh,Vector3(0, 0, 1), max, min, speed, torque))

hip2 = GameObject(position=Vector3(0, 13, 0), size=Vector3(.5, .5, .5), name="hip2").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(hip1,Vector3(1, 0, 0), max, min, speed, torque))

config = Config(
        obs_dim=142,
        action_dim_continuous=5,
        rollout_steps = 10000,
        device="cpu",
        max_steps=400,
        best_model_path="model.pt",
)

Academy.setup_trainer(config)
Academy.load_trained_model(model="model.pt")

leg = GameObject(size=Vector3(), children=[feet, mount, servo1, servo2, calf, mount2, knee, thigh, hip1, hip2]).add_component(MoveToGoal(goal))

scene = [floor, wall1, wall2, wall3, wall4, leg, goal, cam]

Core.run(scene, tick=1/120)