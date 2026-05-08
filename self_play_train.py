from arena import Arena


def main(rounds=20):
    arena = Arena()
    names = list(arena.agents)
    scores = {name: 0.0 for name in names}

    for round_idx in range(rounds):
        print(f"\n=== ROUND {round_idx + 1} ===")
        for i, a_name in enumerate(names):
            for b_name in names[i + 1:]:
                score = arena.match(a_name, b_name)
                print(f"{a_name} vs {b_name} = {score}")
                scores[a_name] += score
                scores[b_name] -= score

    print("\n=== FINAL SCORES ===")
    for name, score in sorted(scores.items(), key=lambda item: -item[1]):
        print(f"{name}: {score}")


if __name__ == "__main__":
    main()
