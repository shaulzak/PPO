import math
import random

from bereshit import Vector3
from bereshit.addons.PPO import Agent
from bereshit.addons.PPO.essentials import GoalReach


class MoveToGoal(Agent):
    def __init__(self, goal, control_dt):
        super(MoveToGoal, self).__init__()
        self.success = 0
        self.episodes = 0
        self.speed = 120
        self.goal = goal
        # Time between Update() calls (the scriptRefreshRate passed to Core.run).
        # The engine passes the physics tick as dt, which is much shorter.
        self.control_dt = control_dt

    def attach(self, parent):
        self.bodyParts = parent.get_all_children_physics()
        for bodypart in self.bodyParts:
            if bodypart.name != "feet":
                bodypart.add_component(GoalReach(self))

        self.servos = parent.search_by_component("Servo")
        self.hip1 = parent.search_by_name("hip2")[0]
        self.hip2 = parent.search_by_name("hip2")[1]
        self.hip = parent.search_by_name("hip_bone")[0]

    def get_distance(self):
        # World positions: the two legs have different parents, so local positions are not comparable.
        dis1 = (self.hip1.transform.position - self.goal.transform.position).magnitude()
        dis2 = (self.hip2.transform.position - self.goal.transform.position).magnitude()
        dis3 = (self.hip.transform.position - self.goal.transform.position).magnitude()
        dis = max(dis1,max(dis2, dis3))

        return dis

    def OnEpisodeBegin(self):
        self.parent.reset_to_default()
        self.goal.reset_to_default()
        start_offset = Vector3(random.uniform(-10, 10), 0, random.uniform(-10, 10))
        self.parent.transform.local_position += start_offset
        # Goal at hip height, 8-12 units from the robot in a random direction.
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(8, 12)
        self.goal.transform.local_position = start_offset + Vector3(math.cos(angle) * distance, 14, math.sin(angle) * distance)
        self.start_dis = self.get_distance()
        self.min_distance = self.start_dis
        self.episodes += 1

    def add_observations(self):
        for body_part in self.bodyParts:
            self.add_observation(body_part.transform.position)
            self.add_observation(body_part.Rigidbody.velocity)
            self.add_observation(body_part.Rigidbody.angular_velocity)
            self.add_observation(body_part.transform.quaternion)

            if body_part in self.servos:
                self.add_observation(body_part.ServoController.get_target_angle())

        self.add_observation(self.goal.transform.position)

    def move(self, actions, dt):
        for i, servo in enumerate(self.servos):
            servo.ServoController.move(actions[i] * self.speed, dt)

    def addRewardByDistance(self):
        distance = self.get_distance()

        if distance < self.min_distance:
            reward = (self.min_distance - distance) / self.start_dis
            self.add_reward(reward)
            self.min_distance = distance

    def check(self):
        if self.hip.transform.position.y >= 16:
            self.add_reward(-3)
            self.end_episode()

    def Update(self, dt):
        if self.process_end_request():
            return
        self.check()
        self.add_observations()
        actions = self.get_continuous_actions()
        self.move(actions, self.control_dt)
        self.addRewardByDistance()
        if self.trainer.learn_if_ready():
            if self.episodes != 0:
                print(self.success/self.episodes)
            self.episodes = 0
            self.success = 0
