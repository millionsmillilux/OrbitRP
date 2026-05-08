def compute_reward(prev, current):

    reward = 0

    # 1. PLANET CONTROL (MOST IMPORTANT SIGNAL)
    reward += 3 * (current["my_planets"] - prev["my_planets"])

    # 2. ENEMY PLANET DAMAGE
    reward += 2 * (prev["enemy_planets"] - current["enemy_planets"])

    # 3. SHIP ADVANTAGE
    reward += 0.01 * (current["my_ships"] - prev["my_ships"])

    # 4. SHIP EFFICIENCY (PUNISH WASTE)
    reward -= 0.005 * current["ships_lost"]

    # 5. STAGNATION PENALTY (prevents doing nothing)
    if current["actions_taken"] < 1:
        reward -= 1

    # 6. WIN BONUS (VERY IMPORTANT)
    if current["winner"] == "me":
        reward += 50

    return reward