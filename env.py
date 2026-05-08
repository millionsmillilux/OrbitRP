import math
import copy

FLEET_SPEED = 2.0
CAPTURE_RADIUS = 5.0
MAX_STEPS = 200
WIN_BONUS = 1000.0


class OrbitEnv:
    def __init__(self, num_agents=3, max_steps=MAX_STEPS, debug=False):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.debug = debug
        self.reset()

    # -----------------------------
    # RESET
    # -----------------------------
    def reset(self):
        self.step_count = 0
        self.done = False

        self.planets = self._init_planets()
        self.fleets = []
        self.stats = self._empty_stats()
        self.prev_metrics = self._metrics()

        return self._get_all_obs()

    def _empty_stats(self):
        return {
            "invalid_actions": 0,
            "actions_clamped": 0,
            "fleets_created": 0,
            "ownership_changes": 0,
            "reward_variance": 0.0,
        }

    # -----------------------------
    # MAP
    # -----------------------------
    def _init_planets(self):
        return [
            {"id": 0, "owner": 0, "x": 20.0, "y": 50.0, "ships": 50, "prod": 2},
            {"id": 1, "owner": 1, "x": 80.0, "y": 50.0, "ships": 50, "prod": 2},
            {"id": 2, "owner": -1, "x": 50.0, "y": 20.0, "ships": 30, "prod": 3},
            {"id": 3, "owner": -1, "x": 50.0, "y": 80.0, "ships": 30, "prod": 3},
        ]

    # -----------------------------
    # OBS
    # -----------------------------
    def _get_obs(self, pid):
        return {
            "player": pid,
            "step": self.step_count,
            "planets": copy.deepcopy(self.planets),
            "fleets": copy.deepcopy(self.fleets),
            "features": self._get_features(pid),
        }

    def _get_features(self, pid):
        return {
            "player": pid,
            "planets": [
                [p["x"], p["y"], p["owner"], p["ships"], p["prod"]]
                for p in self.planets
            ],
            "fleets": [
                [f["x"], f["y"], f["vx"], f["vy"], f["owner"], f["ships"]]
                for f in self.fleets
            ],
        }

    def _get_all_obs(self):
        return [self._get_obs(i) for i in range(self.num_agents)]

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, actions):
        prev_metrics = self.prev_metrics
        self.stats = self._empty_stats()

        if actions is None:
            self.stats["invalid_actions"] += self.num_agents
            actions = [[] for _ in range(self.num_agents)]

        # Normalize malformed action batches to no-ops.
        if not isinstance(actions, list) or len(actions) != self.num_agents:
            self.stats["invalid_actions"] += self.num_agents
            actions = [[] for _ in range(self.num_agents)]

        for pid in range(self.num_agents):
            self._apply_actions(pid, actions[pid])

        self._simulate_fleets()
        self._produce()

        self.step_count += 1
        self.done = self.step_count >= self.max_steps
        self.prev_metrics = self._metrics()
        rewards = self._compute_rewards(prev_metrics, self.prev_metrics)
        self.stats["reward_variance"] = self._variance(rewards)

        return self._get_all_obs(), rewards, self.done, {"stats": copy.deepcopy(self.stats)}

    # -----------------------------
    # ACTIONS
    # -----------------------------
    def _apply_actions(self, pid, actions):
        if not actions:
            return

        for a in actions:
            if not isinstance(a, (list, tuple)) or len(a) != 3:
                self.stats["invalid_actions"] += 1
                continue

            src, target_or_angle, ships = a

            try:
                src = int(src)
                ships = float(ships)
            except (TypeError, ValueError):
                self.stats["invalid_actions"] += 1
                continue

            if src < 0 or src >= len(self.planets):
                self.stats["invalid_actions"] += 1
                continue

            p = self.planets[src]

            if p["owner"] != pid:
                self.stats["invalid_actions"] += 1
                continue

            angle = self._action_angle(src, target_or_angle)
            if angle is None:
                self.stats["invalid_actions"] += 1
                continue

            original_ships = ships
            ships = int(max(0, min(ships, p["ships"])))
            if ships != original_ships:
                self.stats["actions_clamped"] += 1
            if ships <= 0:
                self.stats["invalid_actions"] += 1
                continue

            p["ships"] -= ships

            self.fleets.append({
                "owner": pid,
                "source": src,
                "x": p["x"],
                "y": p["y"],
                "vx": math.cos(angle) * FLEET_SPEED,
                "vy": math.sin(angle) * FLEET_SPEED,
                "ships": ships,
            })
            self.stats["fleets_created"] += 1

    def _action_angle(self, src, target_or_angle):
        if isinstance(target_or_angle, int):
            target_id = target_or_angle
            if 0 <= target_id < len(self.planets) and target_id != src:
                source = self.planets[src]
                target = self.planets[target_id]
                return math.atan2(target["y"] - source["y"], target["x"] - source["x"])
            return None

        try:
            return float(target_or_angle)
        except (TypeError, ValueError):
            return None

    # -----------------------------
    # SIMULATION
    # -----------------------------
    def _simulate_fleets(self):
        remaining = []
        arrivals = {p["id"]: [] for p in self.planets}

        for f in self.fleets:
            f["x"] += f["vx"]
            f["y"] += f["vy"]

            hit_planet_id = None

            for p in self.planets:
                if p["id"] == f.get("source"):
                    continue
                if self._dist(p, f) < CAPTURE_RADIUS:
                    hit_planet_id = p["id"]
                    break

            if hit_planet_id is None:
                remaining.append(f)
            else:
                arrivals[hit_planet_id].append(f)

        self.fleets = remaining

        for p in self.planets:
            if arrivals[p["id"]]:
                self._resolve_arrivals(p, arrivals[p["id"]])

    def _resolve_arrivals(self, planet, fleets):
        old_owner = planet["owner"]
        strengths = {planet["owner"]: int(planet["ships"])}

        for f in fleets:
            owner = int(f["owner"])
            strengths[owner] = strengths.get(owner, 0) + int(f["ships"])

        ranked = sorted(strengths.items(), key=lambda item: item[1], reverse=True)
        top_owner, top_ships = ranked[0]
        second_ships = ranked[1][1] if len(ranked) > 1 else 0

        if top_ships == second_ships:
            planet["owner"] = -1
            planet["ships"] = 0
            return

        planet["owner"] = top_owner
        planet["ships"] = top_ships - second_ships
        if planet["owner"] != old_owner:
            self.stats["ownership_changes"] += 1

    # -----------------------------
    # PRODUCTION
    # -----------------------------
    def _produce(self):
        for p in self.planets:
            if p["owner"] >= 0:
                p["ships"] += p["prod"]

    # -----------------------------
    # REWARD
    # -----------------------------
    def _compute_rewards(self, prev, current):
        rewards = [0.0 for _ in range(self.num_agents)]

        for pid in range(self.num_agents):
            rewards[pid] += 30.0 * current[pid]["owned_planets"]
            rewards[pid] += 8.0 * current[pid]["prod"]
            rewards[pid] += 0.2 * (current[pid]["ships"] - prev[pid]["ships"])
            rewards[pid] += 0.4 * max(0.0, prev[pid]["enemy_ships"] - current[pid]["enemy_ships"])

        if self.done:
            best_owned = max(m["owned_planets"] for m in current)
            winners = [pid for pid, m in enumerate(current) if m["owned_planets"] == best_owned]
            if len(winners) == 1:
                for pid in range(self.num_agents):
                    rewards[pid] += WIN_BONUS if pid == winners[0] else -WIN_BONUS

        return rewards

    def _metrics(self):
        metrics = []
        total_ships = sum(p["ships"] for p in self.planets if p["owner"] >= 0)

        for pid in range(self.num_agents):
            owned = [p for p in self.planets if p["owner"] == pid]
            ships = sum(p["ships"] for p in owned)
            prod = sum(p["prod"] for p in owned)
            metrics.append({
                "owned_planets": len(owned),
                "ships": ships,
                "prod": prod,
                "enemy_ships": total_ships - ships,
            })

        return metrics

    def _variance(self, values):
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((value - mean) ** 2 for value in values) / len(values)

    # -----------------------------
    # UTILS
    # -----------------------------
    def _dist(self, p, f):
        return math.hypot(p["x"] - f["x"], p["y"] - f["y"])
