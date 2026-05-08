import math
import copy

FLEET_SPEED = 2.0
CAPTURE_RADIUS = 5.0
MAX_STEPS = 200


class OrbitEnv:
    def __init__(self, num_agents=3, max_steps=MAX_STEPS):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.reset()

    # -----------------------------
    # RESET
    # -----------------------------
    def reset(self):
        self.step_count = 0
        self.done = False

        self.planets = self._init_planets()
        self.fleets = []

        return self._get_all_obs()

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
        }

    def _get_all_obs(self):
        return [self._get_obs(i) for i in range(self.num_agents)]

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, actions):
        if actions is None:
            actions = [[] for _ in range(self.num_agents)]

        # HARD FIX: ensure structure
        if not isinstance(actions, list) or len(actions) != self.num_agents:
            actions = [[] for _ in range(self.num_agents)]

        for pid in range(self.num_agents):
            self._apply_actions(pid, actions[pid])

        self._simulate_fleets()
        self._produce()

        self.step_count += 1
        self.done = self.step_count >= self.max_steps

        return self._get_all_obs(), self._compute_rewards(), self.done, {}

    # -----------------------------
    # ACTIONS
    # -----------------------------
    def _apply_actions(self, pid, actions):
        if not actions:
            return

        for a in actions:
            if not isinstance(a, (list, tuple)) or len(a) != 3:
                continue

            src, angle, ships = a

            try:
                src = int(src)
                angle = float(angle)
                ships = float(ships)
            except:
                continue

            if src < 0 or src >= len(self.planets):
                continue

            p = self.planets[src]

            if p["owner"] != pid:
                continue

            ships = int(min(ships, p["ships"]))
            if ships <= 0:
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
    def _compute_rewards(self):
        rewards = [0.0 for _ in range(self.num_agents)]

        for p in self.planets:
            if p["owner"] >= 0:
                rewards[p["owner"]] += p["ships"] + (p["prod"] * 5)

        return rewards

    # -----------------------------
    # UTILS
    # -----------------------------
    def _dist(self, p, f):
        return math.hypot(p["x"] - f["x"], p["y"] - f["y"])
