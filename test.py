from env import OrbitEnv
from agents.wrapper import agents

env = OrbitEnv()
obs = env.reset()

for step_num in range(20):
    actions = [agents[i](obs[i]) for i in range(3)]
    obs, reward, done, _ = env.step(actions)
    print(f"Step {step_num}: rewards={reward}, done={done}")
    if done:
        break