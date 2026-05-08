from env import OrbitEnv
from agents.wrapper import agents


def main():
    env = OrbitEnv()
    obs = env.reset()

    for step_num in range(20):
        actions = [agents[i](obs[i]) for i in range(3)]
        obs, rewards, done, info = env.step(actions)
        print(f"step={step_num} rewards={rewards} done={done} stats={info['stats']}")
        if done:
            break


if __name__ == "__main__":
    main()
