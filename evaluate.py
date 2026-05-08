from arena import Arena


def main():
    arena = Arena()
    names = list(arena.agents)
    results = {}

    for a_name in names:
        for b_name in names:
            if a_name == b_name:
                continue
            results[f"{a_name} vs {b_name}"] = arena.match(a_name, b_name)

    print("\n=== AGENT RANKINGS ===")
    for name, score in results.items():
        print(f"{name}: {score}")


if __name__ == "__main__":
    main()
