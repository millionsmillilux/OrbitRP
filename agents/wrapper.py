from agents import agent_a, agent_b, agent_c

def safe(action):
    if action is None:
        return []
    if not isinstance(action, list):
        return []
    return action


def act_a(obs):
    return safe(agent_a.act(obs))


def act_b(obs):
    return safe(agent_b.act(obs))


def act_c(obs):
    return safe(agent_c.act(obs))


agents = [act_a, act_b, act_c]