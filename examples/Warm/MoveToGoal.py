import math
import random

import keyboard

from bereshit import Vector3
from bereshit.addons.PPO import Agent
from bereshit.addons.PPO.essentials import GoalReach


class MoveToGoal(Agent):
    def __init__(self, goal):
        super(MoveToGoal, self).__init__()
        self.goal = goal
        self.speed = 120
        self.start_dis = 0
        self.episodes = 0
        self.success = 0
        self.min_distance = 0

    def attach(self, parent):
        self.bodyParts = parent.get_all_children_physics()
        for bodypart in self.bodyParts:
            bodypart.add_component(GoalReach(self))


        self.servos = parent.search_by_component("Servo")
        self.head = parent.search_by_name("head")[0]

    def get_distance(self):
        return (self.head.transform.local_position - self.goal.transform.local_position).magnitude()

    def OnEpisodeBegin(self):
        self.parent.reset_to_default()
        self.goal.reset_to_default()
        self.parent.transform.local_position += Vector3(random.uniform(-10, 10), 0, random.uniform(-10, 10))
        self.goal.transform.local_position += Vector3(random.uniform(-10, 10), 0, random.uniform(-10, 10))
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
        direction = (
                self.goal.transform.position
                - self.head.transform.position
        )

        self.add_observation(direction.normalized())
        self.add_observation(direction.magnitude() / 30.0)

    def move(self, actions, dt):
        for i, servo in enumerate(self.servos):
            servo.ServoController.move(actions[i] * self.speed, dt)

    def addRewardByDistance(self):
        distance = self.get_distance()

        if distance < self.min_distance:
            reward = (self.min_distance - distance) / self.start_dis
            self.add_reward(reward)
            self.min_distance = distance


    def add_shaping_reward(self):
        to_goal = (
                self.goal.transform.local_position
                - self.head.transform.local_position
        )

        direction = to_goal.normalized()

        velocity = self.head.Rigidbody.velocity

        # Reward velocity pointing toward goal
        forward_speed = velocity.dot(direction)

        self.add_reward(0.002 * forward_speed)

        # Tiny time penalty
        self.add_reward(-0.0005)

    def get_Actions(self):
        if keyboard.is_pressed("h"):
            return [1,1,1]
        if keyboard.is_pressed("g"):
            return [-1,-1,-1]
        return [0,0,0]

    def Update(self, dt):
        if self.process_end_request():
            return
        self.add_observations()
        actions = self.get_continuous_actions()
        # actions = self.get_Actions()
        self.move(actions, dt)
        self.addRewardByDistance()
        # self.add_reward(-0.0001)
        if self.trainer.learn_if_ready():
            if self.episodes != 0:
                print(self.success/self.episodes)
            self.episodes = 0
            self.success = 0



