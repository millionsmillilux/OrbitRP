from agents.wrapper import agents
from env import OrbitEnv


def play_match(a_idx, b_idx):
    env = OrbitEnv()
    obs = env.reset()

    rewards = [0, 0, 0]

    for _ in range(200):
        actions = [[] for _ in range(3)]

        actions[a_idx] = agents[a_idx](obs[a_idx])
        actions[b_idx] = agents[b_idx](obs[b_idx])

        obs, reward, done, _ = env.step(actions)

        rewards = reward

        if done:
            break

    return rewards[a_idx] - rewards[b_idx]


def main():
    scores = [0, 0, 0]

    pairs = [(0,1), (0,2), (1,2)]

    for _ in range(20):
        print("\n=== ROUND ===")
        for a, b in pairs:
            score = play_match(a, b)
            print(f"{a} vs {b} = {score}")
            scores[a] += score
            scores[b] -= score

    print("\n=== FINAL SCORES ===")
    print("A:", scores[0])
    print("B:", scores[1])
    print("C:", scores[2])


if __name__ == "__main__":
    main()