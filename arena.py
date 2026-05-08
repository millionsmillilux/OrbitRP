from agents.wrapper import agents as default_agents
from env import OrbitEnv


class Arena:
    def __init__(self, agents_dict=None, max_steps=None):
        if agents_dict is None:
            agents_dict = {
                "a": default_agents[0],
                "b": default_agents[1],
                "c": default_agents[2],
            }

        self.agents = agents_dict
        self.max_steps = max_steps

    def match(self, name_a, name_b):
        env = OrbitEnv(max_steps=self.max_steps or 200)
        obs = env.reset()
        done = False
        rewards = [0.0, 0.0, 0.0]

        agent_a = self.agents[name_a]
        agent_b = self.agents[name_b]

        while not done:
            actions = [[] for _ in range(env.num_agents)]
            actions[0] = agent_a(obs[0])
            actions[1] = agent_b(obs[1])

            obs, rewards, done, _ = env.step(actions)

        return rewards[0] - rewards[1]
