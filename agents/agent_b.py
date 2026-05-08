from agents.agent_a import agent as agent_impl

def act(obs):
    return agent_impl(obs)

# Expose as 'agent' for wrapper compatibility
agent = act