from bereshit import Component

class GoalReach(Component):
    def __init__(self, agent):
        super(GoalReach, self).__init__()
        self.agent = agent

    # Collision callbacks run inside the physics step, so only request the end here;
    # the agent ends the episode from its Update() via process_end_request().
    def OnCollisionEnter(self, Collision):
        if Collision.other.parent.get_component("Wall"):
            self.agent.request_end_episode(-1)
        elif Collision.other.parent.get_component("Goal"):
            if self.agent.request_end_episode(1):
                self.agent.success += 1
