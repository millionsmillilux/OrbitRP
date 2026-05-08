import math


def act(obs):
    player = int(obs["player"])
    planets = obs["planets"]
    moves = []

    owned = [p for p in planets if p["owner"] == player]
    targets = [p for p in planets if p["owner"] != player]

    for src in owned:
        available = int(src["ships"]) - 18
        if available <= 0:
            continue
        target = min(
            targets,
            key=lambda p: (
                p["owner"] == -1,
                p["ships"],
                math.hypot(p["x"] - src["x"], p["y"] - src["y"]),
            ),
            default=None,
        )
        if target is None:
            continue
        send = min(available, max(int(target["ships"]) + 1, 20))
        if send > 0:
            moves.append([src["id"], target["id"], send])

    return moves
