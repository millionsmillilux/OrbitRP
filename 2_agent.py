"""
Orbit Wars — NOVA v1  (Next-gen Orbital Victory Architecture)
=============================================================

Architecture overview  (priority pipeline, executed every turn)
---------------------------------------------------------------
  0.  Parse & classify game state
  1.  Desperation — critical-loss all-in
  2.  Emergency evacuation — save ships from overwhelmed planets
  3.  Counter-attack windows — strike weakened enemy planets
  4.  Multi-tier reinforcement (immediate + preemptive)
  5.  Snipe racing — reach neutrals before enemy fleets do
  6.  Early-game orbital-rush (first RUSH_TURNS turns)
  7.  Winning-mode crush — aggressive elimination finish
  8.  Orbital-sync multi-assault — pool planets with timed arrivals
  9.  Production-denial targeting — destroy top enemy producers
 10.  Satellite-denial — preempt forward enemy staging planets
 11.  Global greedy attack — worldwide two-pass assignment
 12.  Adaptive fleet splitting — claim multiple neutrals cheaply
 13.  Consolidation — forward surplus rear ships to the front

Unique advanced systems
-----------------------
  OrbitalPhaseAnalyzer
    - For every orbiting planet, pre-computes the next solar-clear
      launch windows and optimal aim-lead angles for each source.
    - Identifies "orbital alignment events" where two targets pass
      near the same region simultaneously → single-fleet double-cap.

  InfluenceMap (12×12 grid over 100×100 board)
    - Fills a grid with min-ETA reachability from each side.
    - Used to identify contested neutrals, safe zones, and to
      prioritise targets in the enemy's "sphere of influence."

  EconomicProjector
    - Projects production income and ship counts N turns forward,
      accounting for in-transit fleets, planned attacks, and production.
    - Detects "economic crossover" — the turn we overtake enemy production
      — to time aggressive investment vs defensive holding.

  ThreatSimulator
    - Simulates all known enemy fleets forward to find which of our
      planets will be attacked in the next 50 turns.
    - Detects "fleet cascade" — enemy sends follow-up fleets to the
      same target to break our defenders.

  EnemyBehaviorModel
    - Counts enemy actions per turn, preferred target types, aggression
      radius, and fleet size patterns to predict next enemy moves.

  ProductionDenialScorer
    - Computes a "denial value" for each enemy planet = production lost
      × remaining game turns — weighted above standard target score.

Physics (sourced from Kaggle orbit_wars environment)
----------------------------------------------------
  Fleet speed  : 1.0 + (max_speed−1) × (log(n)/log(1000))^1.5
  Planet orbit : future_angle = current_angle + omega × t
  Sun kill-zone: path within SUN_RADIUS=10 of (50,50) is fatal
  Intercept    : hit when |spd×t − d| ≤ planet.radius  (both bounds)
"""

import math
import time
from collections import defaultdict, namedtuple

# ===========================================================================
# Named tuples — mirror kaggle_environments.envs.orbit_wars.orbit_wars
# ===========================================================================

try:
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
except ImportError:
    Planet = namedtuple(
        "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
    )
    Fleet = namedtuple(
        "Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"]
    )

# ===========================================================================
# Board constants
# ===========================================================================

BOARD_SIZE          = 100.0
CENTER              = 50.0
SUN_RADIUS          = 10.0
_SUN_R_SQ           = SUN_RADIUS * SUN_RADIUS
ROTATION_THRESHOLD  = 50.0

# ===========================================================================
# Strategy constants
# ===========================================================================

MIN_SEND              = 11    # absolute hard floor — never violated
EMERGENCY_MIN         = MIN_SEND
GARRISON_BASE         = 2
MAX_SEARCH_T          = 90    # board diagonal / min speed ≈ 70; 90 has margin
CAPTURE_EXTRA_NEUTRAL = 3
CAPTURE_EXTRA_ENEMY   = 6
W_PROD                = 12.0
W_ETA                 = 1.3
W_COST                = 0.9
ENEMY_BONUS           = 1.5
SAFETY_RATIO          = 1.2
CONSOLIDATE_THRESHOLD = 26
PREEMPT_LOOKAHEAD     = 28
FRONT_DIST            = 35.0
REAR_DIST             = 55.0
ARRIVAL_TOL           = 5     # multi-assault: max ETA difference (turns)
RUSH_TURNS            = 40
TIME_BUDGET           = 0.68  # hard bail-out to stay within 1 s actTimeout
INFLUENCE_GRID_DIM    = 12    # resolution of influence map (12×12)
ECON_LOOKAHEAD        = 40    # turns ahead for economic projection
DENIAL_TURNS_REMAIN   = 200   # turns weight for production-denial scoring
SATELLITE_RADIUS      = 30.0  # preempt neutrals within this dist of enemies

# ===========================================================================
# Observation parsing
# ===========================================================================

def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _cfg(cfg, key, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _to_planet(raw):
    if isinstance(raw, Planet):
        return raw
    if isinstance(raw, (list, tuple)):
        return Planet(*raw)
    return Planet(
        id         = int(getattr(raw, "id",         0)),
        owner      = int(getattr(raw, "owner",      -1)),
        x          = float(getattr(raw, "x",        0.0)),
        y          = float(getattr(raw, "y",        0.0)),
        radius     = float(getattr(raw, "radius",   1.0)),
        ships      = int(getattr(raw, "ships",      0)),
        production = int(getattr(raw, "production", 1)),
    )


def _to_fleet(raw):
    if isinstance(raw, Fleet):
        return raw
    if isinstance(raw, (list, tuple)):
        return Fleet(*raw)
    return Fleet(
        id             = int(getattr(raw, "id",             0)),
        owner          = int(getattr(raw, "owner",          -1)),
        x              = float(getattr(raw, "x",            0.0)),
        y              = float(getattr(raw, "y",            0.0)),
        angle          = float(getattr(raw, "angle",        0.0)),
        from_planet_id = int(getattr(raw, "from_planet_id", -1)),
        ships          = int(getattr(raw, "ships",          0)),
    )


def parse_obs(obs, cfg):
    step       = int(_get(obs, "step")               or 0)
    player     = int(_get(obs, "player")             or 0)
    omega      = float(_get(obs, "angular_velocity") or 0.0)
    ship_speed = float(_cfg(cfg, "shipSpeed",        6.0))
    planets    = [_to_planet(p) for p in list(_get(obs, "planets")          or [])]
    fleets     = [_to_fleet(f)  for f in list(_get(obs, "fleets")           or [])]
    comet_ids  = set(int(c)     for c  in list(_get(obs, "comet_planet_ids") or []))
    return step, player, omega, ship_speed, planets, fleets, comet_ids

# ===========================================================================
# Geometry helpers
# ===========================================================================

def dist2(ax, ay, bx, by):
    dx = ax - bx; dy = ay - by
    return dx * dx + dy * dy


def dist(ax, ay, bx, by):
    dx = ax - bx; dy = ay - by
    return math.sqrt(dx * dx + dy * dy)


def _seg_min_dist_sq(px, py, ax, ay, bx, by):
    """Squared min-distance from point to segment — no sqrt overhead."""
    dx = bx - ax; dy = by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        ex = px - ax; ey = py - ay
        return ex * ex + ey * ey
    t  = ((px - ax) * dx + (py - ay) * dy) / l2
    t  = max(0.0, min(1.0, t))
    ex = px - (ax + t * dx); ey = py - (ay + t * dy)
    return ex * ex + ey * ey


def crosses_sun(fx, fy, tx, ty):
    return _seg_min_dist_sq(CENTER, CENTER, fx, fy, tx, ty) < _SUN_R_SQ


def angle_between(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)

# ===========================================================================
# Fleet physics
# ===========================================================================

def fleet_speed(n_ships, ship_speed_max):
    """
    Exact formula from orbit_wars source:
      speed = 1 + (max−1) × (log(n)/log(1000))^1.5  capped at max.

    Notable values (max=6):
       11 ships  →  2.02 u/t
       43 ships  →  3.00 u/t
      100 ships  →  4.00 u/t
      999 ships  →  6.00 u/t
    """
    if n_ships <= 1:
        return 1.0
    s = 1.0 + (ship_speed_max - 1.0) * (math.log(n_ships) / math.log(1000)) ** 1.5
    return min(s, ship_speed_max)


def turns_to_reach(distance, n_ships, ship_speed_max):
    """Ceiling of distance / speed."""
    spd = fleet_speed(n_ships, ship_speed_max)
    return max(1, int(math.ceil(distance / spd))) if spd > 0 else MAX_SEARCH_T

# ===========================================================================
# Orbital mechanics
# ===========================================================================

def _orbit_params(planet):
    """(is_orbiting, orbit_radius, base_angle)."""
    dx = planet.x - CENTER; dy = planet.y - CENTER
    r  = math.sqrt(dx * dx + dy * dy)
    return (r + planet.radius) < ROTATION_THRESHOLD, r, math.atan2(dy, dx)


def planet_pos_at(planet, t_ahead, omega):
    """
    Predict (x, y) of planet t_ahead turns from now.
    Static planets (orbit_r + radius ≥ 50) return current position.
    """
    dx = planet.x - CENTER; dy = planet.y - CENTER
    r  = math.sqrt(dx * dx + dy * dy)
    if (r + planet.radius) >= ROTATION_THRESHOLD:
        return planet.x, planet.y
    a = math.atan2(dy, dx) + omega * t_ahead
    return CENTER + r * math.cos(a), CENTER + r * math.sin(a)


def planet_velocity(planet, omega):
    """
    Instantaneous velocity vector (vx, vy) of an orbiting planet.
    Tangent to the circular orbit, perpendicular to the radius vector.
    Returns (0, 0) for static planets.
    """
    dx = planet.x - CENTER; dy = planet.y - CENTER
    r  = math.sqrt(dx * dx + dy * dy)
    if (r + planet.radius) >= ROTATION_THRESHOLD or r < 1e-6:
        return 0.0, 0.0
    # Tangential velocity: magnitude = omega × r, direction = perpendicular to (dx,dy)
    mag = omega * r
    # Perpendicular unit vector (rotate 90° anti-clockwise): (-dy/r, dx/r)
    return -mag * dy / r, mag * dx / r

# ===========================================================================
# Intercept calculation
#
# Both-bound contact condition: |spd×t − d| ≤ radius  (NOT just spd×t ≥ d−r)
# This prevents false positives when the fleet massively overshoots.
# Sun-blocked paths: increment t so orbiting planets rotate into a clear lane.
# Static planets behind the sun are unreachable → return (None, None).
# ===========================================================================

def find_intercept(fx, fy, target_planet, n_ships, ship_speed_max, omega):
    """
    Earliest turn T at which a fleet of n_ships from (fx,fy) intercepts planet.
    Returns (eta_turns, aim_angle) or (None, None).
    """
    spd = fleet_speed(n_ships, ship_speed_max)
    r   = target_planet.radius

    orbiting, orb_r, base_angle = _orbit_params(target_planet)
    if not orbiting:
        tx_s, ty_s = target_planet.x, target_planet.y

    for t in range(1, MAX_SEARCH_T + 1):
        if orbiting:
            a  = base_angle + omega * t
            tx = CENTER + orb_r * math.cos(a)
            ty = CENTER + orb_r * math.sin(a)
        else:
            tx, ty = tx_s, ty_s

        ddx = tx - fx; ddy = ty - fy
        d   = math.sqrt(ddx * ddx + ddy * ddy)

        # Both-bounds contact check (fixes original one-sided bug)
        if abs(spd * t - d) <= r:
            if crosses_sun(fx, fy, tx, ty):
                continue
            return t, math.atan2(ddy, ddx)

    return None, None


def find_intercept_two_pass(fx, fy, tgt, available,
                             ship_speed_max, omega,
                             enemy_inc, friend_inc, my_player):
    """
    Two-pass intercept: rough ETA at max speed → derive send count → precise ETA.
    Returns (eta, angle, send) or (None, None, None).
    """
    eta_rough, _ = find_intercept(fx, fy, tgt, available, ship_speed_max, omega)
    if eta_rough is None:
        return None, None, None

    def_rough  = garrison_at_arrival(tgt, eta_rough, enemy_inc, friend_inc, my_player)
    is_neutral = (tgt.owner == -1)
    needed     = def_rough + (CAPTURE_EXTRA_NEUTRAL if is_neutral else CAPTURE_EXTRA_ENEMY)
    send       = max(MIN_SEND, min(available, needed))
    if send > available:
        return None, None, None

    eta, angle = find_intercept(fx, fy, tgt, send, ship_speed_max, omega)
    if eta is None:
        return None, None, None

    return eta, angle, send

# ===========================================================================
# Fleet destination tracking  (angular-deviation — scale-free)
# ===========================================================================

def fleet_dest_planet(fleet, planets, max_angle_deg=25.0):
    """
    Planet most aligned with fleet heading within max_angle_deg.
    Returns None if nothing qualifies.
    """
    dx      = math.cos(fleet.angle)
    dy      = math.sin(fleet.angle)
    max_dev = math.radians(max_angle_deg)
    best, best_dev = None, max_dev

    for p in planets:
        vx = p.x - fleet.x; vy = p.y - fleet.y
        d  = math.sqrt(vx * vx + vy * vy)
        if d < 1e-6:
            continue
        proj = vx * dx + vy * dy
        if proj <= 0:
            continue
        perp = abs(vx * dy - vy * dx)
        dev  = math.asin(min(1.0, perp / d))
        if dev < best_dev:
            best_dev = dev; best = p

    return best


def build_incoming_maps(fleets, planets, my_player):
    """
    enemy_inc[pid]  = total enemy ships en route to planet pid
    friend_inc[pid] = total friendly ships en route to planet pid
    """
    enemy_inc  = defaultdict(int)
    friend_inc = defaultdict(int)
    for fl in fleets:
        dest = fleet_dest_planet(fl, planets)
        if dest is None:
            continue
        if fl.owner == my_player:
            friend_inc[dest.id] += fl.ships
        else:
            enemy_inc[dest.id] += fl.ships
    return dict(enemy_inc), dict(friend_inc)

# ===========================================================================
# Ship forecasting
# ===========================================================================

def garrison_at_arrival(planet, eta, enemy_inc, friend_inc, my_player):
    """
    Estimated defending ship count when our fleet arrives at `eta` turns from now.

    Neutral: static until captured; if enemy_inc > planet.ships, simulate capture
             and post-capture enemy production growth.
    Enemy  : grows by production + incoming enemy reinforcements.
    Ours   : grows by production + friendly inbound − enemy inbound.
    """
    if planet.owner == -1:
        ei = enemy_inc.get(planet.id, 0)
        if ei > planet.ships:
            flip_t   = max(1, eta // 2)
            post_cap = ei - planet.ships + planet.production * (eta - flip_t)
            return max(0, int(post_cap))
        return max(0, planet.ships)

    if planet.owner == my_player:
        fi = friend_inc.get(planet.id, 0)
        ei = enemy_inc.get(planet.id, 0)
        return max(0, planet.ships + planet.production * eta + fi - ei)

    # Enemy
    ei = enemy_inc.get(planet.id, 0)
    return max(0, planet.ships + planet.production * eta + ei)


def ships_needed_to_capture(defending, is_neutral):
    buf = CAPTURE_EXTRA_NEUTRAL if is_neutral else CAPTURE_EXTRA_ENEMY
    return defending + buf

# ===========================================================================
# Safety guard
# ===========================================================================

def safe_to_attack(src_planet, send_ships, enemy_inc):
    remaining = src_planet.ships - send_ships
    incoming  = enemy_inc.get(src_planet.id, 0)
    required  = max(GARRISON_BASE, int(math.ceil(incoming * SAFETY_RATIO)))
    return remaining >= required


def garrison_for_planet(planet, role, enemy_inc, friend_inc):
    """Adaptive garrison by strategic role."""
    ei  = enemy_inc.get(planet.id, 0)
    fi  = friend_inc.get(planet.id, 0)
    net = max(0, ei - fi)
    if role == "front":
        return max(GARRISON_BASE + 4, net + 3)
    elif role == "mid":
        return max(GARRISON_BASE + 1, net + 1)
    else:
        return max(GARRISON_BASE, net + 1)

# ===========================================================================
# Fleet ETA estimation
# ===========================================================================

def estimate_fleet_eta(fleet, dest_planet, ship_speed_max):
    spd = fleet_speed(fleet.ships, ship_speed_max)
    d   = dist(fleet.x, fleet.y, dest_planet.x, dest_planet.y)
    return max(1, int(math.ceil(d / spd)))

# ===========================================================================
# Game state analysis
# ===========================================================================

def game_phase(planets, my_player):
    n  = len(planets)
    if n == 0:
        return "early"
    nf = sum(1 for p in planets if p.owner == -1) / n
    return "early" if nf > 0.55 else ("mid" if nf > 0.20 else "late")


def phase_weights(phase, production_lead=0.0):
    """(W_PROD, W_ETA, W_COST, ENEMY_BONUS) tuned per phase and econ gap."""
    if phase == "early":
        return 16.0, 1.0, 0.7, 1.3
    elif phase == "mid":
        eta_w = max(0.9, 1.3 - production_lead * 0.05)
        return 13.0, eta_w, 0.9, 1.6
    else:
        eta_w = max(1.1, 1.6 - production_lead * 0.03)
        return 10.0, eta_w, 0.8, 2.2


def strength_ratio(all_planets, all_fleets, my_player):
    my_s = (sum(p.ships for p in all_planets if p.owner == my_player)
            + sum(f.ships for f in all_fleets if f.owner == my_player))
    tot  = (sum(p.ships for p in all_planets if p.owner not in (-1,))
            + sum(f.ships for f in all_fleets if f.owner not in (-1,)))
    return my_s / tot if tot > 0 else 0.5


def production_gap(all_planets, my_player):
    """my_prod − max_enemy_prod."""
    my_prod = sum(p.production for p in all_planets if p.owner == my_player)
    ep = defaultdict(int)
    for p in all_planets:
        if p.owner not in (-1, my_player):
            ep[p.owner] += p.production
    max_ep = max(ep.values()) if ep else 0
    return my_prod - max_ep


def desperation_mode(all_planets, all_fleets, my_player):
    my_s = (sum(p.ships for p in all_planets if p.owner == my_player)
            + sum(f.ships for f in all_fleets if f.owner == my_player))
    en_s = (sum(p.ships for p in all_planets if p.owner not in (-1, my_player))
            + sum(f.ships for f in all_fleets if f.owner not in (-1, my_player)))
    return en_s > 0 and my_s < en_s * 0.30


def winning_mode(all_planets, all_fleets, my_player):
    return strength_ratio(all_planets, all_fleets, my_player) >= 0.62

# ===========================================================================
# Planet classification
# ===========================================================================

def classify_planets(my_planets, all_planets, my_player):
    """Assign 'front' / 'mid' / 'rear' to each owned planet."""
    enemies = [p for p in all_planets if p.owner not in (-1, my_player)]
    roles   = {}
    for p in my_planets:
        if not enemies:
            roles[p.id] = "rear"; continue
        nearest = min(dist(p.x, p.y, e.x, e.y) for e in enemies)
        if nearest <= FRONT_DIST:
            roles[p.id] = "front"
        elif nearest <= REAR_DIST:
            roles[p.id] = "mid"
        else:
            roles[p.id] = "rear"
    return roles

# ===========================================================================
# Threat analysis
# ===========================================================================

def immediate_threats(my_planets, enemy_inc, friend_inc):
    out = {}
    for p in my_planets:
        ei  = enemy_inc.get(p.id, 0)
        fi  = friend_inc.get(p.id, 0)
        net = ei - (p.ships + fi)
        if net > 0:
            out[p.id] = net
    return out


def preemptive_threats(my_planets, enemy_planets_list, my_player,
                       ship_speed_max, n_lookahead=PREEMPT_LOOKAHEAD):
    out   = {}
    max_d = ship_speed_max * n_lookahead
    for p in my_planets:
        urgency = 0.0
        for ep in enemy_planets_list:
            d = dist(p.x, p.y, ep.x, ep.y)
            if d > max_d:
                continue
            eta_t   = max(1, int(d / ship_speed_max))
            ep_str  = ep.ships + ep.production * eta_t
            our_gar = p.ships
            if ep_str > our_gar * 1.5:
                urgency = max(urgency, (ep_str - our_gar) / (eta_t + 1))
        if urgency > 0.0:
            out[p.id] = urgency
    return out

# ===========================================================================
# *** NEW *** OrbitalPhaseAnalyzer
#
# For each orbiting planet, pre-computes which turns (within MAX_SEARCH_T)
# the direct path from a source is NOT blocked by the sun.  This gives us
# a set of "clear windows" we can aim for and avoids wasting time in the
# inner intercept loop re-discovering sun blockages.
#
# Also computes the "alignment bonus" — when the planet's orbital velocity
# is moving TOWARD the source, the leading angle is shallower and the fleet
# needs fewer turns.  We return an eta_bonus (negative = actually faster).
# ===========================================================================

class OrbitalPhaseAnalyzer:
    """
    Pre-computes orbital intercept windows for every planet–source pair.

    Usage:
        opa = OrbitalPhaseAnalyzer(planets, omega, ship_speed_max)
        windows = opa.clear_windows(src_planet, target_planet, n_ships)
        # windows: list of (t, angle) pairs where interception is sun-safe

    The first entry in windows is the best (earliest) intercept.
    """

    def __init__(self, planets, omega, ship_speed_max):
        self.omega          = omega
        self.ship_speed_max = ship_speed_max
        # Pre-cache orbit params for all planets
        self._params = {p.id: _orbit_params(p) for p in planets}

    def clear_windows(self, fx, fy, target_planet, n_ships, max_t=None):
        """
        Returns list of (t, angle) for all sun-safe intercept turns up to max_t.
        Empty list means no intercept possible.
        """
        if max_t is None:
            max_t = MAX_SEARCH_T
        spd = fleet_speed(n_ships, self.ship_speed_max)
        r   = target_planet.radius
        omega = self.omega
        orbiting, orb_r, base_angle = self._params[target_planet.id]

        results = []
        for t in range(1, max_t + 1):
            if orbiting:
                a  = base_angle + omega * t
                tx = CENTER + orb_r * math.cos(a)
                ty = CENTER + orb_r * math.sin(a)
            else:
                tx, ty = target_planet.x, target_planet.y

            ddx = tx - fx; ddy = ty - fy
            d   = math.sqrt(ddx * ddx + ddy * ddy)
            if abs(spd * t - d) <= r and not crosses_sun(fx, fy, tx, ty):
                results.append((t, math.atan2(ddy, ddx)))
                if not orbiting:
                    break  # static planet — all future t also valid; just return first

        return results

    def best_intercept(self, fx, fy, target_planet, n_ships):
        """Returns (t, angle) for earliest safe intercept, or (None, None)."""
        wins = self.clear_windows(fx, fy, target_planet, n_ships, MAX_SEARCH_T)
        return wins[0] if wins else (None, None)

    def orbital_approach_bonus(self, fx, fy, target_planet):
        """
        Returns a signed bonus in (−1, +1) range:
          −1 → planet is rushing toward source (shorter ETA, good)
          +1 → planet is fleeing away from source (longer ETA, bad)
        Used as a soft priority tiebreaker for target selection.
        """
        vx, vy = planet_velocity(target_planet, self.omega)
        # Vector from planet to source
        sx = fx - target_planet.x; sy = fy - target_planet.y
        d  = math.sqrt(sx * sx + sy * sy)
        if d < 1e-6 or (vx == 0 and vy == 0):
            return 0.0
        # Cosine of angle between planet velocity and source direction
        # Negative = moving toward source (approach bonus is negative = better)
        v_mag = math.sqrt(vx * vx + vy * vy)
        return (vx * sx + vy * sy) / (v_mag * d)

    def find_alignment_time(self, planet_a, planet_b, horizon=MAX_SEARCH_T):
        """
        Find the earliest future turn when planet_a and planet_b are within
        2 × (radius_a + radius_b) of each other — an "orbital alignment event."
        Returns the turn number, or None.
        """
        threshold_sq = (2 * (planet_a.radius + planet_b.radius)) ** 2
        for t in range(1, horizon + 1):
            ax, ay = planet_pos_at(planet_a, t, self.omega)
            bx, by = planet_pos_at(planet_b, t, self.omega)
            if dist2(ax, ay, bx, by) <= threshold_sq:
                return t
        return None

# ===========================================================================
# *** NEW *** InfluenceMap
#
# A 12×12 grid over the 100×100 board.  Each cell stores which side (my_player
# or enemy) can reach it faster, and by how many turns.
# Used for:
#   • Identifying "contested" neutrals (influence delta ≤ 5 turns).
#   • Deprioritising neutrals deep in enemy influence.
#   • Prioritising forward neutral planets in our influence zone.
# ===========================================================================

class InfluenceMap:
    """
    Grid-based spatial influence map for Orbit Wars.

    Each cell (i, j) stores:
        influence[i][j] = (controlling_owner, turn_advantage)
    where turn_advantage > 0 means we get there N turns faster.
    """

    def __init__(self, my_planets, enemy_planets, ship_speed_max, dim=INFLUENCE_GRID_DIM):
        self.dim  = dim
        self.cell = BOARD_SIZE / dim
        self._map = {}  # (i, j) → (owner, advantage)
        self._build(my_planets, enemy_planets, ship_speed_max)

    def _cell_centre(self, i, j):
        return (i + 0.5) * self.cell, (j + 0.5) * self.cell

    def _build(self, my_planets, enemy_planets, ship_speed_max):
        dim = self.dim
        # Approximate speed for a standard fleet of MIN_SEND ships
        spd = fleet_speed(MIN_SEND, ship_speed_max)
        for i in range(dim):
            for j in range(dim):
                cx, cy = self._cell_centre(i, j)
                my_eta  = min(
                    (dist(p.x, p.y, cx, cy) / spd for p in my_planets),
                    default=9999.0
                )
                en_eta  = min(
                    (dist(p.x, p.y, cx, cy) / spd for p in enemy_planets),
                    default=9999.0
                ) if enemy_planets else 9999.0
                adv = en_eta - my_eta   # positive = we get there first
                owner = "mine" if adv >= 0 else "enemy"
                self._map[(i, j)] = (owner, adv)

    def _planet_cell(self, planet):
        i = min(self.dim - 1, int(planet.x / self.cell))
        j = min(self.dim - 1, int(planet.y / self.cell))
        return i, j

    def advantage_at(self, planet):
        """Returns signed advantage for this planet's grid cell (positive = we lead)."""
        cell = self._planet_cell(planet)
        return self._map.get(cell, ("mine", 0.0))[1]

    def is_contested(self, planet, threshold=6.0):
        """True if neither side has a clear advantage (within threshold turns)."""
        return abs(self.advantage_at(planet)) <= threshold

    def is_ours(self, planet, threshold=0.0):
        return self.advantage_at(planet) > threshold

    def is_enemy(self, planet, threshold=0.0):
        return self.advantage_at(planet) < -threshold

# ===========================================================================
# *** NEW *** EconomicProjector
#
# Projects ship count and production for both sides N turns forward.
# Used to:
#   • Detect "economic crossover" — when we surpass enemy production.
#   • Decide whether to spend ships now (attack) or accumulate (defend).
#   • Score production-denial targets more accurately.
# ===========================================================================

class EconomicProjector:
    """Projects fleet economies N turns forward."""

    def __init__(self, planets, fleets, my_player, ship_speed_max):
        self.planets       = planets
        self.fleets        = fleets
        self.my_player     = my_player
        self.ship_speed_max = ship_speed_max

    def _side_snapshot(self, owner_check):
        """Current ships on planets + fleets for a side."""
        p_ships = sum(p.ships for p in self.planets if owner_check(p.owner))
        f_ships = sum(f.ships for f in self.fleets  if owner_check(f.owner))
        prod    = sum(p.production for p in self.planets if owner_check(p.owner))
        return p_ships + f_ships, prod

    def project_strength(self, turns):
        """
        Returns (my_strength, enemy_strength) after `turns` turns,
        assuming no new actions (conservative lower bound for us).
        """
        my_now, my_prod = self._side_snapshot(lambda o: o == self.my_player)
        en_now, en_prod = self._side_snapshot(
            lambda o: o not in (-1, self.my_player)
        )
        return my_now + my_prod * turns, en_now + en_prod * turns

    def crossover_turn(self, horizon=ECON_LOOKAHEAD):
        """
        Turn at which our strength overtakes enemy (or None if never within horizon).
        Useful: if crossover is near, consider holding ships rather than attacking.
        """
        my_now, my_prod = self._side_snapshot(lambda o: o == self.my_player)
        en_now, en_prod = self._side_snapshot(
            lambda o: o not in (-1, self.my_player)
        )
        if my_now >= en_now and my_prod >= en_prod:
            return 0   # already ahead on both dimensions
        if en_prod <= my_prod:
            return None  # we grow faster, crossover has passed
        # Ships: my_now + my_prod*t = en_now + en_prod*t → t = (en-my)/(my_prod-en_prod)
        delta_prod = my_prod - en_prod
        if delta_prod <= 0:
            return None
        t = (en_now - my_now) / delta_prod
        return int(math.ceil(t)) if 0 < t <= horizon else None

    def production_denial_value(self, enemy_planet, turns_remaining=DENIAL_TURNS_REMAIN):
        """
        Value gained by capturing this enemy planet =
          production lost by enemy × remaining turns.
        """
        return enemy_planet.production * turns_remaining

    def should_accumulate(self, turns=15):
        """
        True if holding ships for `turns` more turns puts us decisively ahead.
        Signals conservative play when we have an economic advantage building.
        """
        ms_now, _ = self._side_snapshot(lambda o: o == self.my_player)
        ms_fut, en_fut = self.project_strength(turns)
        return ms_fut > en_fut * 1.4 and ms_now < en_fut * 0.9

# ===========================================================================
# *** NEW *** EnemyBehaviorModel
#
# Tracks enemy fleet patterns across turns to predict next moves.
# Persists across calls using module-level state.
# ===========================================================================

_behavior_state = {
    "enemy_action_count": defaultdict(int),   # owner → total fleets seen
    "preferred_prod":     defaultdict(float),  # owner → avg target production
    "aggression_radius":  defaultdict(list),   # owner → list of fleet distances
    "last_fleet_ids":     set(),               # fleet IDs seen last turn
    "fleet_size_history": defaultdict(list),   # owner → list of fleet sizes
}


def update_enemy_model(fleets, planets, my_player):
    """
    Update behavior state with current fleet observations.
    Call once per turn before planning.
    """
    state   = _behavior_state
    pid_map = {p.id: p for p in planets}

    current_ids = set()
    for fl in fleets:
        if fl.owner == my_player:
            continue
        current_ids.add(fl.id)
        if fl.id in state["last_fleet_ids"]:
            continue   # already counted

        # New enemy fleet detected this turn
        owner = fl.owner
        state["enemy_action_count"][owner] += 1
        state["fleet_size_history"][owner].append(fl.ships)

        # Estimate aggression radius from launch planet
        src = pid_map.get(fl.from_planet_id)
        if src:
            d = dist(src.x, src.y, fl.x, fl.y)
            state["aggression_radius"][owner].append(d)

    state["last_fleet_ids"] = current_ids


def enemy_aggression_score(enemy_owner):
    """
    Returns estimated aggression level 0..1 for an enemy player.
    High = sends many large fleets far away.
    """
    state   = _behavior_state
    sizes   = state["fleet_size_history"][enemy_owner]
    radii   = state["aggression_radius"][enemy_owner]
    if not sizes:
        return 0.5   # neutral assumption
    avg_size   = sum(sizes) / len(sizes)
    avg_radius = sum(radii) / len(radii) if radii else 30.0
    # Normalize: 100 ships at 50 units = 1.0
    return min(1.0, (avg_size / 100.0) * (avg_radius / 50.0))

# ===========================================================================
# *** NEW *** ThreatSimulator
#
# Simulates ALL current enemy fleets forward T turns to find which of our
# planets will be attacked, and by how many ships.
# Much more accurate than just checking enemy_inc (which only sees current
# positions and doesn't model fleet travel time properly).
# ===========================================================================

class ThreatSimulator:
    """
    Forward-simulates enemy fleet arrivals on our planets.

    Results: {planet_id: [(arrival_turn, fleet_ships), ...]}
    """

    def __init__(self, enemy_fleets, planets, my_player, ship_speed_max, omega):
        self.planet_map      = {p.id: p for p in planets}
        self.my_planets      = {p.id for p in planets if p.owner == my_player}
        self.ship_speed_max  = ship_speed_max
        self.omega           = omega
        self._arrivals       = self._simulate(enemy_fleets, planets)

    def _simulate(self, enemy_fleets, planets):
        arrivals = defaultdict(list)
        for fl in enemy_fleets:
            dest = fleet_dest_planet(fl, planets)
            if dest is None or dest.id not in self.my_planets:
                continue
            spd = fleet_speed(fl.ships, self.ship_speed_max)
            d   = dist(fl.x, fl.y, dest.x, dest.y)
            eta = max(1, int(math.ceil(d / spd)))
            arrivals[dest.id].append((eta, fl.ships))
        return dict(arrivals)

    def arrivals_at(self, planet_id):
        """List of (eta, ships) for enemy fleets heading to planet_id."""
        return self._arrivals.get(planet_id, [])

    def total_incoming(self, planet_id, window=None):
        """Total enemy ships arriving at planet within `window` turns."""
        lst = self.arrivals_at(planet_id)
        if window is None:
            return sum(s for _, s in lst)
        return sum(s for t, s in lst if t <= window)

    def is_safe(self, planet_id, garrison, window=10):
        """True if garrison can survive all arrivals within window turns."""
        return self.total_incoming(planet_id, window) < garrison

    def cascade_threat(self, planet_id, garrison):
        """
        Detect fleet cascade: multiple waves arriving successively that
        will overwhelm even a reinforced garrison.
        Returns (True, first_breach_turn) or (False, None).
        """
        lst = sorted(self.arrivals_at(planet_id))
        running_garrison = garrison
        for eta, ships in lst:
            running_garrison -= ships
            if running_garrison <= 0:
                return True, eta
        return False, None

# ===========================================================================
# *** NEW *** ProductionDenialScorer
#
# Adds a "denial bonus" to enemy planet scores:
#   denial_value = production_per_turn × turns_remaining_in_game
# This makes destroying high-production enemy planets worth much more
# than the base scoring formula captures.
# ===========================================================================

def production_denial_score(planet, turns_remaining, projector):
    """
    Extra score for capturing this enemy planet due to production denial.
    Blended with regular target_score in plan_global_attacks.
    """
    if planet.owner == -1:
        return 0.0
    return projector.production_denial_value(planet, turns_remaining)


def target_score_with_denial(target, eta, net_defending,
                              wp, we, wc, eb,
                              prod_lead=0.0,
                              denial_bonus=0.0):
    """
    Extended target scoring: standard formula + production-denial bonus.
    denial_bonus is pre-computed from ProductionDenialScorer.
    """
    prod        = max(1, target.production)
    bonus       = eb if target.owner not in (-1,) else 1.0
    prod_factor = prod * wp * bonus * (1.0 + max(0.0, prod_lead) * 0.02 * prod)

    if eta <= 0:
        return -1e9

    base_score  = prod_factor / (max(1, eta) ** we * max(1.0, net_defending) ** wc)
    denial_factor = denial_bonus * 0.0001   # scale denial into same magnitude as base
    return base_score + denial_factor

# ===========================================================================
# Global Coordinator
# ===========================================================================

class GlobalCoordinator:
    """Tracks committed ships per destination planet this planning cycle."""

    def __init__(self):
        self._committed = defaultdict(int)

    def committed(self, planet_id):
        return self._committed[planet_id]

    def add(self, planet_id, ships):
        self._committed[planet_id] += ships

    def register_action(self, dest_id, ships):
        self.add(dest_id, ships)

    def is_covered(self, target, defending):
        needed = ships_needed_to_capture(defending, target.owner == -1)
        return self._committed[target.id] >= needed

# ===========================================================================
# Target ranking
# ===========================================================================

def rank_targets(all_planets, my_player, comet_ids, winning=False):
    """Priority-ordered attackable planets: enemy → neutral → comets."""
    enemies  = []
    neutrals = []
    comets_l = []

    for p in all_planets:
        if p.owner == my_player:
            continue
        if p.id in comet_ids:
            comets_l.append(p)
        elif p.owner == -1:
            neutrals.append(p)
        else:
            enemies.append(p)

    enemies.sort(key=lambda p: -p.production)
    neutrals.sort(key=lambda p: -p.production)
    comets_l.sort(key=lambda p: p.ships)

    return enemies + neutrals + comets_l

# ===========================================================================
# Safety: detect if our own fleet might be "poisoned" by en-route enemy
# (enemy fleet arrives at target planet just before ours, turning neutral→enemy)
# ===========================================================================

def fleet_destination_risk(target_planet, our_eta, enemy_inc, enemy_fleets,
                            planets, ship_speed_max):
    """
    Estimate risk that target flips ownership before our fleet arrives.
    Returns True if an enemy fleet will likely arrive before us and we haven't
    accounted for the increased defender count.
    """
    for fl in enemy_fleets:
        dest = fleet_dest_planet(fl, planets)
        if dest is None or dest.id != target_planet.id:
            continue
        spd = fleet_speed(fl.ships, ship_speed_max)
        d   = dist(fl.x, fl.y, dest.x, dest.y)
        en_eta = max(1, int(math.ceil(d / spd)))
        if en_eta < our_eta:
            return True  # enemy gets there first
    return False

# ===========================================================================
# Desperation tactic
# ===========================================================================

def plan_desperation(my_planets, all_planets, omega, ship_speed_max,
                     my_player, deadline):
    """All-in: send everything from every available planet toward easiest targets."""
    targets = sorted(
        [p for p in all_planets if p.owner != my_player],
        key=lambda p: p.ships
    )
    if not targets:
        return []

    actions   = []
    used_srcs = set()

    for tgt in targets:
        if time.time() >= deadline:
            break
        for src in sorted(my_planets, key=lambda p: -p.ships):
            if src.id in used_srcs:
                continue
            available = src.ships - 1
            if available < MIN_SEND:
                continue
            eta, angle = find_intercept(src.x, src.y, tgt, available, ship_speed_max, omega)
            if eta is None:
                continue
            actions.append([src.id, float(angle), int(available)])
            used_srcs.add(src.id)
            break
        if len(used_srcs) >= max(1, len(my_planets) - 1):
            break

    return actions

# ===========================================================================
# Emergency evacuation
# ===========================================================================

def plan_emergency_evacuation(my_planets, planet_map, imm_threats,
                               enemy_inc, friend_inc, my_player,
                               omega, ship_speed_max, evac_blocked, deadline):
    """Save ships from planets that are overwhelmingly outgunned."""
    actions = []

    for pid, threat in imm_threats.items():
        if time.time() >= deadline:
            break
        src = planet_map.get(pid)
        if src is None or src.id in evac_blocked:
            continue
        if threat < src.ships:
            continue  # can hold; evacuation not needed

        safe_dests = [
            p for p in my_planets
            if p.id != pid and enemy_inc.get(p.id, 0) == 0
        ]
        if not safe_dests:
            continue

        refuge   = min(safe_dests, key=lambda p: dist2(src.x, src.y, p.x, p.y))
        evacuate = src.ships - 1

        if evacuate < EMERGENCY_MIN:
            continue

        eta, angle = find_intercept(src.x, src.y, refuge, evacuate, ship_speed_max, omega)
        if eta is None:
            continue

        actions.append([src.id, float(angle), int(evacuate)])
        evac_blocked.add(src.id)

    return actions

# ===========================================================================
# Counter-attack windows
# ===========================================================================

def find_counter_attack_windows(all_planets, all_fleets, my_player,
                                 my_planets, enemy_inc, friend_inc,
                                 ship_speed_max, omega, blocked_srcs,
                                 coord, deadline):
    """
    Hit enemy planets that just sent large fleets out and are temporarily weak.
    Detection: outbound ships from enemy planet >> current garrison.
    """
    actions  = []
    outbound = defaultdict(int)
    for fl in all_fleets:
        if fl.owner == my_player:
            continue
        outbound[fl.from_planet_id] += fl.ships

    for ep in all_planets:
        if ep.owner in (-1, my_player):
            continue
        if time.time() >= deadline:
            break

        out_from_ep = outbound.get(ep.id, 0)
        if out_from_ep < MIN_SEND:
            continue
        if ep.ships > 25:
            continue
        if coord.is_covered(ep, ep.ships):
            continue

        for src in sorted(my_planets, key=lambda p: dist2(p.x, p.y, ep.x, ep.y)):
            if src.id in blocked_srcs:
                continue
            role      = "front"   # counter-attack always from front
            gar       = garrison_for_planet(src, role, enemy_inc, friend_inc)
            available = src.ships - gar
            if available < MIN_SEND:
                continue

            eta, angle = find_intercept(src.x, src.y, ep, available, ship_speed_max, omega)
            if eta is None:
                continue

            defending = garrison_at_arrival(ep, eta, enemy_inc, friend_inc, my_player)
            needed    = ships_needed_to_capture(defending, False)
            send      = max(MIN_SEND, min(available, needed))
            if send < MIN_SEND or send > available:
                continue
            if not safe_to_attack(src, send, enemy_inc):
                continue

            eta2, angle2 = find_intercept(src.x, src.y, ep, send, ship_speed_max, omega)
            if eta2 is None:
                continue

            actions.append([src.id, float(angle2), int(send)])
            coord.register_action(ep.id, send)
            blocked_srcs.add(src.id)
            break

    return actions

# ===========================================================================
# Reinforcement planning (multi-tier)
# ===========================================================================

def plan_reinforcements(my_planets, planet_map, imm_threats, pre_threats,
                        enemy_inc, friend_inc, my_player,
                        omega, ship_speed_max, planet_roles, deadline):
    """
    Tier 1 — immediate: planets with enemy fleets already inbound.
    Tier 2 — preemptive: planets predicted under threat in next 28 turns.
    Returns (actions, blocked_srcs, dest_ids).
    """
    actions   = []
    used_srcs = set()
    dest_ids  = set()

    all_threats = {}
    for pid, t in imm_threats.items():
        all_threats[pid] = ("immediate", t)
    for pid, u in pre_threats.items():
        if pid not in all_threats:
            all_threats[pid] = ("preemptive", u)

    sorted_threats = sorted(
        all_threats.items(),
        key=lambda kv: (0 if kv[1][0] == "immediate" else 1, -kv[1][1])
    )

    for pid, (tier, threat_val) in sorted_threats:
        if time.time() >= deadline:
            break
        dest = planet_map.get(pid)
        if dest is None:
            continue

        donors = sorted(
            [p for p in my_planets
             if p.id != pid
             and p.id not in used_srcs
             and p.ships - GARRISON_BASE >= MIN_SEND],
            key=lambda p: dist2(p.x, p.y, dest.x, dest.y)
        )

        for donor in donors:
            if time.time() >= deadline:
                break
            gar       = garrison_for_planet(donor, planet_roles.get(donor.id, "rear"),
                                            enemy_inc, friend_inc)
            available = donor.ships - gar
            send      = min(available, int(threat_val) + CAPTURE_EXTRA_ENEMY)
            send      = max(send, MIN_SEND)
            if send < MIN_SEND or send > available:
                continue

            eta, angle = find_intercept(donor.x, donor.y, dest, send, ship_speed_max, omega)
            if eta is None:
                continue

            actions.append([donor.id, float(angle), int(send)])
            used_srcs.add(donor.id)
            dest_ids.add(dest.id)
            break

    return actions, used_srcs, dest_ids

# ===========================================================================
# Snipe racing — capture neutrals before enemy fleets do
# ===========================================================================

def plan_snipe_captures(my_planets, all_planets, enemy_fleets,
                        enemy_inc, friend_inc, my_player,
                        omega, ship_speed_max, blocked_srcs,
                        planet_map, coord, deadline):
    """
    Identify neutral planets where an enemy fleet is en route and we can
    arrive first (or tie) — then commit our own fleet to win the race.
    """
    actions  = []
    new_srcs = set()

    # Build ETA map for enemy fleets to neutral planets
    neutral_enemy_eta = defaultdict(list)  # planet_id → [enemy_eta, ...]
    for fl in enemy_fleets:
        dest = fleet_dest_planet(fl, all_planets)
        if dest is None or dest.owner != -1:
            continue
        spd = fleet_speed(fl.ships, ship_speed_max)
        d   = dist(fl.x, fl.y, dest.x, dest.y)
        en_eta = max(1, int(math.ceil(d / spd)))
        neutral_enemy_eta[dest.id].append(en_eta)

    if not neutral_enemy_eta:
        return [], set()

    # Sort neutral races by enemy ETA ascending (soonest to flip = most urgent)
    race_targets = sorted(
        [(pid, min(etas)) for pid, etas in neutral_enemy_eta.items()
         if planet_map.get(pid) and not coord.is_covered(planet_map[pid], planet_map[pid].ships)],
        key=lambda x: x[1]
    )

    used_tgts = set()

    for pid, en_eta in race_targets:
        if time.time() >= deadline:
            break
        tgt = planet_map.get(pid)
        if tgt is None or tgt.id in used_tgts:
            continue

        for src in sorted(
            [p for p in my_planets if p.id not in blocked_srcs and p.id not in new_srcs],
            key=lambda p: dist2(p.x, p.y, tgt.x, tgt.y)
        ):
            if time.time() >= deadline:
                break
            available = src.ships - GARRISON_BASE
            if available < MIN_SEND:
                continue

            eta, angle = find_intercept(src.x, src.y, tgt, available, ship_speed_max, omega)
            if eta is None or eta > en_eta + 2:
                continue  # we arrive too late to matter

            defending = garrison_at_arrival(tgt, eta, enemy_inc, friend_inc, my_player)
            needed    = ships_needed_to_capture(defending, True)
            send      = max(MIN_SEND, min(available, needed))
            if send < MIN_SEND or send > available:
                continue
            if not safe_to_attack(src, send, enemy_inc):
                continue

            eta2, angle2 = find_intercept(src.x, src.y, tgt, send, ship_speed_max, omega)
            if eta2 is None:
                continue

            actions.append([src.id, float(angle2), int(send)])
            coord.register_action(tgt.id, send)
            new_srcs.add(src.id)
            used_tgts.add(tgt.id)
            break

    return actions, new_srcs

# ===========================================================================
# Early-game orbital rush
# ===========================================================================

def plan_early_rush(my_planets, all_planets, enemy_inc, friend_inc,
                    my_player, omega, ship_speed_max,
                    blocked_srcs, coord, deadline,
                    influence_map=None):
    """
    First RUSH_TURNS turns: sprint to 3 nearest high-production neutrals.
    Influence-map aware: deprioritise neutrals deep in enemy territory.
    """
    actions  = []
    new_srcs = set()

    rush_targets = sorted(
        [p for p in all_planets
         if p.owner == -1
         and not coord.is_covered(p, p.ships)],
        key=lambda p: (
            # If we have influence map, deprioritise enemy-influenced targets
            1 if (influence_map and influence_map.is_enemy(p, threshold=8.0)) else 0,
            -p.production
        )
    )
    if not rush_targets:
        return [], set()

    assigned_targets = set()

    for src in sorted(my_planets, key=lambda p: -(p.ships - GARRISON_BASE)):
        if src.id in blocked_srcs or src.id in new_srcs:
            continue
        if time.time() >= deadline:
            break

        available = src.ships - GARRISON_BASE
        if available < MIN_SEND:
            continue

        best_tgt  = None
        best_eta  = MAX_SEARCH_T + 1

        for tgt in rush_targets:
            if tgt.id in assigned_targets:
                continue

            eta, angle = find_intercept(src.x, src.y, tgt, available, ship_speed_max, omega)
            if eta is None:
                continue

            defending = garrison_at_arrival(tgt, eta, enemy_inc, friend_inc, my_player)
            needed    = ships_needed_to_capture(defending, True)
            send      = max(MIN_SEND, min(available, needed))
            if send > available:
                continue

            if eta < best_eta:
                best_eta  = eta
                best_tgt  = (tgt, angle, send)

        if best_tgt is None:
            continue

        tgt, angle, send = best_tgt
        eta2, angle2 = find_intercept(src.x, src.y, tgt, send, ship_speed_max, omega)
        if eta2 is None:
            continue
        if not safe_to_attack(src, send, enemy_inc):
            continue

        actions.append([src.id, float(angle2), int(send)])
        coord.register_action(tgt.id, send)
        assigned_targets.add(tgt.id)
        new_srcs.add(src.id)

    return actions, new_srcs

# ===========================================================================
# *** NEW *** Orbital-sync multi-assault
#
# Finds the future time T when the target orbiting planet is in an "optimal
# attack position" — i.e., when the most source planets can reach it with
# nearly identical ETAs (within ARRIVAL_TOL turns), enabling a coordinated
# simultaneous strike that overwhelms any single-source limit.
#
# Algorithm:
#   1. For each candidate hard enemy target, scan t=1..MAX_SEARCH_T.
#   2. At each t, compute how many of our free planets can intercept at t ± TOL.
#   3. Pick the t that maximises total sender count × total ships available.
#   4. Emit launch orders adjusted so all arrive within ARRIVAL_TOL turns of t.
# ===========================================================================

def plan_orbital_sync_assault(my_planets, all_planets, enemy_inc, friend_inc,
                               coord, used_srcs, my_player,
                               omega, ship_speed_max, phase, prod_lead,
                               deadline, opa):
    """
    Coordinated orbital-sync attack on strongly defended orbiting planets.
    Returns list of [from_id, angle, ships].
    """
    actions = []

    hard_targets = [
        p for p in all_planets
        if p.owner not in (-1, my_player)
        and p.ships > 25
        and not coord.is_covered(p, p.ships)
        and _orbit_params(p)[0]   # only orbiting planets benefit from sync
    ]
    if not hard_targets:
        return []

    free = [
        p for p in my_planets
        if p.id not in used_srcs
        and p.ships - GARRISON_BASE >= MIN_SEND
    ]
    if len(free) < 2:
        return []

    committed_here = set()

    for tgt in sorted(hard_targets, key=lambda p: -p.production):
        if time.time() >= deadline:
            break
        if coord.is_covered(tgt, tgt.ships):
            continue

        best_t      = None
        best_senders = []
        best_power   = 0

        # Scan future turns for best alignment moment
        orbiting, orb_r, base_angle = _orbit_params(tgt)

        for t in range(1, MAX_SEARCH_T + 1):
            if time.time() >= deadline:
                break
            a  = base_angle + omega * t
            tx = CENTER + orb_r * math.cos(a)
            ty = CENTER + orb_r * math.sin(a)

            senders_at_t = []
            total_ships  = 0

            for src in free:
                if src.id in committed_here:
                    continue
                available = src.ships - GARRISON_BASE
                if available < MIN_SEND:
                    continue

                # Check if this source can intercept at t ± ARRIVAL_TOL
                spd = fleet_speed(available, ship_speed_max)
                ddx = tx - src.x; ddy = ty - src.y
                d   = math.sqrt(ddx * ddx + ddy * ddy)

                arrived = False
                for dt in range(-ARRIVAL_TOL, ARRIVAL_TOL + 1):
                    t2 = t + dt
                    if t2 < 1 or t2 > MAX_SEARCH_T:
                        continue
                    a2  = base_angle + omega * t2
                    tx2 = CENTER + orb_r * math.cos(a2)
                    ty2 = CENTER + orb_r * math.sin(a2)
                    ddx2 = tx2 - src.x; ddy2 = ty2 - src.y
                    d2   = math.sqrt(ddx2 * ddx2 + ddy2 * ddy2)
                    if abs(spd * t2 - d2) <= tgt.radius and not crosses_sun(src.x, src.y, tx2, ty2):
                        senders_at_t.append((src, t2, math.atan2(ddy2, ddx2), available))
                        total_ships += available
                        arrived = True
                        break

            if len(senders_at_t) >= 2 and total_ships > best_power:
                best_power   = total_ships
                best_t       = t
                best_senders = senders_at_t

        if best_t is None or len(best_senders) < 2:
            continue

        # Check if combined force is enough to capture
        defending = garrison_at_arrival(tgt, best_t, enemy_inc, friend_inc, my_player)
        needed    = ships_needed_to_capture(defending, False)

        if best_power < needed:
            continue   # combined force still insufficient

        total_committed = 0
        for src, t_launch, angle, available in best_senders:
            if total_committed >= needed:
                break
            my_contrib = min(available, needed - total_committed + CAPTURE_EXTRA_ENEMY)
            my_contrib = max(MIN_SEND, my_contrib)
            if my_contrib > available:
                continue
            if not safe_to_attack(src, my_contrib, enemy_inc):
                continue

            # Re-verify precise intercept at the assigned launch turn
            eta_check, angle_check = find_intercept(
                src.x, src.y, tgt, my_contrib, ship_speed_max, omega
            )
            if eta_check is None:
                continue

            actions.append([src.id, float(angle_check), int(my_contrib)])
            coord.register_action(tgt.id, my_contrib)
            committed_here.add(src.id)
            total_committed += my_contrib

    return actions

# ===========================================================================
# *** NEW *** Production-denial targeting
#
# Identifies the top enemy planets by production rate and scores them
# with an amplified denial bonus, causing the attack planner to prioritise
# them above same-ETA lower-production targets.
# ===========================================================================

def plan_production_denial(my_planets, all_planets, enemy_inc, friend_inc,
                            coord, blocked_srcs, my_player, omega,
                            ship_speed_max, phase, prod_lead,
                            projector, step, max_steps, deadline):
    """
    Focused strikes on the top-2 enemy production planets.
    Uses ProductionDenialScorer to amplify their value.
    Returns (actions, used_srcs).
    """
    actions   = []
    used_srcs = set()
    turns_left = max(1, max_steps - step)

    # Top enemy producers sorted by production desc
    enemy_prods = sorted(
        [p for p in all_planets if p.owner not in (-1, my_player)],
        key=lambda p: -p.production
    )[:2]

    if not enemy_prods:
        return [], set()

    free_planets = [
        p for p in my_planets
        if p.id not in blocked_srcs
        and p.ships - GARRISON_BASE >= MIN_SEND
    ]

    wp, we, wc, eb = phase_weights(phase, prod_lead)

    for tgt in enemy_prods:
        if time.time() >= deadline:
            break
        if coord.is_covered(tgt, tgt.ships):
            continue

        denial_val = production_denial_score(tgt, turns_left, projector)

        best_move  = None
        best_score = -1e18

        for src in sorted(free_planets, key=lambda p: dist2(p.x, p.y, tgt.x, tgt.y)):
            if src.id in used_srcs:
                continue
            if time.time() >= deadline:
                break

            available = src.ships - GARRISON_BASE
            if available < MIN_SEND:
                continue

            eta, angle, send = find_intercept_two_pass(
                src.x, src.y, tgt, available, ship_speed_max, omega,
                enemy_inc, friend_inc, my_player
            )
            if eta is None:
                continue
            if not safe_to_attack(src, send, enemy_inc):
                continue

            defending = garrison_at_arrival(tgt, eta, enemy_inc, friend_inc, my_player)
            net_def   = max(0, defending - coord.committed(tgt.id))

            sc = target_score_with_denial(
                tgt, eta, net_def, wp, we, wc, eb, prod_lead, denial_val
            )

            if sc > best_score:
                best_score = sc
                best_move  = (src.id, float(angle), int(send), src)

        if best_move:
            sid, angle, send, src_obj = best_move
            actions.append([sid, angle, send])
            coord.register_action(tgt.id, send)
            used_srcs.add(sid)

    return actions, used_srcs

# ===========================================================================
# *** NEW *** Satellite-denial
#
# Identifies neutral planets within SATELLITE_RADIUS of enemy-owned planets
# that could serve as forward staging areas for the enemy.  Captures them
# preemptively to deny the enemy that foothold.
# ===========================================================================

def plan_satellite_denial(my_planets, all_planets, enemy_inc, friend_inc,
                           coord, blocked_srcs, my_player, omega,
                           ship_speed_max, deadline, influence_map):
    """
    Capture neutral planets near enemy territory before the enemy can use them
    as forward bases.
    Returns (actions, used_srcs).
    """
    actions   = []
    used_srcs = set()

    enemy_planets = [p for p in all_planets if p.owner not in (-1, my_player)]
    if not enemy_planets:
        return [], set()

    # Find neutral planets close to enemies but in contested/our territory
    satellite_targets = []
    for p in all_planets:
        if p.owner != -1:
            continue
        if coord.is_covered(p, p.ships):
            continue
        # Must be within SATELLITE_RADIUS of some enemy
        near_enemy = min(dist(p.x, p.y, ep.x, ep.y) for ep in enemy_planets)
        if near_enemy > SATELLITE_RADIUS:
            continue
        # Prefer contested zones or ones we can reach faster than enemy
        adv = influence_map.advantage_at(p)
        if adv < -10.0:
            continue  # deep enemy territory — skip
        satellite_targets.append((p, near_enemy, adv))

    # Sort: closest to enemy first (most dangerous to leave uncaptured)
    satellite_targets.sort(key=lambda x: x[1])

    free_planets = [
        p for p in my_planets
        if p.id not in blocked_srcs
        and p.ships - GARRISON_BASE >= MIN_SEND
    ]

    assigned_tgts = set()

    for tgt, near_d, adv in satellite_targets:
        if time.time() >= deadline:
            break
        if tgt.id in assigned_tgts:
            continue

        best_move  = None
        best_eta   = MAX_SEARCH_T + 1

        for src in sorted(free_planets, key=lambda p: dist2(p.x, p.y, tgt.x, tgt.y)):
            if src.id in used_srcs:
                continue
            available = src.ships - GARRISON_BASE
            if available < MIN_SEND:
                continue

            eta, angle = find_intercept(src.x, src.y, tgt, available, ship_speed_max, omega)
            if eta is None or eta >= best_eta:
                continue

            defending = garrison_at_arrival(tgt, eta, enemy_inc, friend_inc, my_player)
            needed    = ships_needed_to_capture(defending, True)
            send      = max(MIN_SEND, min(available, needed))
            if send > available:
                continue
            if not safe_to_attack(src, send, enemy_inc):
                continue

            eta2, angle2 = find_intercept(src.x, src.y, tgt, send, ship_speed_max, omega)
            if eta2 is None:
                continue

            best_eta  = eta2
            best_move = (src.id, float(angle2), int(send), src.id)

        if best_move:
            sid, angle, send, _ = best_move
            actions.append([sid, angle, send])
            coord.register_action(tgt.id, send)
            used_srcs.add(sid)
            assigned_tgts.add(tgt.id)

    return actions, used_srcs

# ===========================================================================
# *** NEW *** Adaptive fleet splitting
#
# When we have several spare ships and multiple neutrals are close together
# and weak (≤15 ships), split our fleet to claim several simultaneously
# instead of sending one large fleet that wastes overkill on each.
# ===========================================================================

def plan_adaptive_split(my_planets, all_planets, enemy_inc, friend_inc,
                         coord, blocked_srcs, my_player, omega,
                         ship_speed_max, phase, deadline):
    """
    Claim multiple cheap neutrals simultaneously by splitting large fleets.
    Only active during 'early' and 'mid' phases.
    Returns (actions, used_srcs).
    """
    if phase == "late":
        return [], set()

    actions   = []
    used_srcs = set()

    cheap_neutrals = sorted(
        [p for p in all_planets
         if p.owner == -1 and p.ships <= 18
         and not coord.is_covered(p, p.ships)],
        key=lambda p: p.ships
    )
    if len(cheap_neutrals) < 2:
        return [], set()

    rich_planets = sorted(
        [p for p in my_planets
         if p.id not in blocked_srcs
         and p.ships - GARRISON_BASE >= MIN_SEND * 3],
        key=lambda p: -(p.ships - GARRISON_BASE)
    )

    assigned_tgts  = set()

    for src in rich_planets:
        if time.time() >= deadline:
            break
        available = src.ships - GARRISON_BASE
        if available < MIN_SEND * 2:
            continue

        # Find the two closest unassigned cheap neutrals
        nearby = [
            n for n in cheap_neutrals
            if n.id not in assigned_tgts and not coord.is_covered(n, n.ships)
        ]
        nearby.sort(key=lambda n: dist2(src.x, src.y, n.x, n.y))

        if len(nearby) < 2:
            continue

        # Try to send to the two closest
        half = available // 2
        if half < MIN_SEND:
            continue

        move_count = 0
        temp_used  = available

        for n in nearby[:2]:
            if temp_used < MIN_SEND:
                break
            send = max(MIN_SEND, n.ships + CAPTURE_EXTRA_NEUTRAL)
            if send > temp_used:
                send = temp_used
            if send < MIN_SEND:
                break

            eta, angle = find_intercept(src.x, src.y, n, send, ship_speed_max, omega)
            if eta is None:
                break

            if not safe_to_attack(src, available - temp_used + send, enemy_inc):
                break

            actions.append([src.id, float(angle), int(send)])
            coord.register_action(n.id, send)
            assigned_tgts.add(n.id)
            temp_used  -= send
            move_count += 1

        if move_count >= 2:
            used_srcs.add(src.id)

    return actions, used_srcs

# ===========================================================================
# Global attack planning — worldwide two-pass greedy assignment
# ===========================================================================

def build_attack_candidates(my_planets, all_planets, enemy_inc, friend_inc,
                             blocked_srcs, my_player, omega, ship_speed_max,
                             comet_ids, phase, prod_lead, winning,
                             projector, step, max_steps, deadline):
    """
    All valid (source, target) attack pairs with two-pass ETA + denial score.
    Returns list sorted by score descending.
    """
    wp, we, wc, eb = phase_weights(phase, prod_lead)
    targets        = rank_targets(all_planets, my_player, comet_ids, winning)
    turns_left     = max(1, max_steps - step)
    candidates     = []

    for src in my_planets:
        if src.id in blocked_srcs:
            continue
        available = src.ships - GARRISON_BASE
        if available < MIN_SEND:
            continue
        if time.time() >= deadline:
            break

        fx, fy = src.x, src.y

        for tgt in targets:
            if tgt.owner == my_player:
                continue
            if time.time() >= deadline:
                break

            eta_rough, _ = find_intercept(fx, fy, tgt, available, ship_speed_max, omega)
            if eta_rough is None:
                continue

            def_rough  = garrison_at_arrival(tgt, eta_rough, enemy_inc, friend_inc, my_player)
            is_neutral = (tgt.owner == -1)
            needed_r   = ships_needed_to_capture(def_rough, is_neutral)

            if not is_neutral and available < needed_r // 2:
                continue

            send = max(MIN_SEND, min(available, needed_r))
            if send < MIN_SEND or send > available:
                continue

            eta, angle = find_intercept(fx, fy, tgt, send, ship_speed_max, omega)
            if eta is None:
                continue

            if not safe_to_attack(src, send, enemy_inc):
                continue

            defending   = garrison_at_arrival(tgt, eta, enemy_inc, friend_inc, my_player)
            net_def     = max(0, defending)
            denial_val  = production_denial_score(tgt, turns_left, projector)

            sc = target_score_with_denial(
                tgt, eta, net_def, wp, we, wc, eb, prod_lead, denial_val
            )

            candidates.append({
                "score":     sc,
                "src_id":    src.id,
                "tgt_id":    tgt.id,
                "tgt":       tgt,
                "angle":     float(angle),
                "send":      int(send),
                "eta":       eta,
                "net_def":   net_def,
                "available": available,
                "src":       src,
            })

    candidates.sort(key=lambda x: -x["score"])
    return candidates


def plan_global_attacks(my_planets, all_planets, enemy_inc, friend_inc,
                        coord, blocked_srcs, my_player, omega,
                        ship_speed_max, comet_ids, phase,
                        prod_lead, winning, projector, step, max_steps, deadline):
    """Globally-coordinated greedy attack: best (source, target) pairs worldwide."""
    candidates = build_attack_candidates(
        my_planets, all_planets, enemy_inc, friend_inc,
        blocked_srcs, my_player, omega, ship_speed_max,
        comet_ids, phase, prod_lead, winning, projector, step, max_steps, deadline
    )

    actions   = []
    used_srcs = set(blocked_srcs)

    for cand in candidates:
        if time.time() >= deadline:
            break

        src_id    = cand["src_id"]
        tgt_id    = cand["tgt_id"]
        tgt       = cand["tgt"]
        angle     = cand["angle"]
        net_def   = cand["net_def"]
        available = cand["available"]

        if src_id in used_srcs:
            continue

        already = coord.committed(tgt_id)
        eff_def = max(0, net_def - already)

        if already > 0 and coord.is_covered(tgt, net_def):
            continue

        needed = ships_needed_to_capture(eff_def, tgt.owner == -1)
        send   = max(MIN_SEND, min(available, needed))
        if send < MIN_SEND or send > available:
            continue

        actions.append([src_id, angle, send])
        coord.register_action(tgt_id, send)
        used_srcs.add(src_id)

    return actions, used_srcs

# ===========================================================================
# Multi-planet coordinated assault (enhanced from aura_v2)
# ===========================================================================

def plan_multi_planet_assaults(my_planets, all_planets, enemy_inc, friend_inc,
                                coord, used_srcs, my_player,
                                omega, ship_speed_max, phase, prod_lead,
                                deadline):
    """
    Pool 2-3 planets against heavily-defended targets.
    ETAs must agree within ARRIVAL_TOL turns.
    """
    _, we, wc, eb = phase_weights(phase, prod_lead)
    actions       = []

    hard = [
        p for p in all_planets
        if p.owner not in (-1, my_player)
        and p.ships > 30
        and not coord.is_covered(p, p.ships)
    ]
    if not hard:
        return []

    free = [
        p for p in my_planets
        if p.id not in used_srcs
        and p.ships - GARRISON_BASE >= MIN_SEND
    ]
    if len(free) < 2:
        return []

    committed_here = set()

    for tgt in sorted(hard, key=lambda p: -p.production):
        if time.time() >= deadline:
            break
        if coord.is_covered(tgt, tgt.ships):
            continue

        # Gather (src, eta, angle, available) for all free planets
        pairs = []
        for src in free:
            if src.id in committed_here:
                continue
            available = src.ships - GARRISON_BASE
            if available < MIN_SEND:
                continue

            eta, angle = find_intercept(
                src.x, src.y, tgt, available, ship_speed_max, omega
            )
            if eta is None:
                continue
            pairs.append((src, eta, angle, available))

        if len(pairs) < 2:
            continue

        # Sort by ETA; pick anchor (median ETA)
        pairs.sort(key=lambda x: x[1])
        anchor_eta = pairs[len(pairs) // 2][1]

        # Select pairs within ARRIVAL_TOL
        synced = [p for p in pairs if abs(p[1] - anchor_eta) <= ARRIVAL_TOL]
        if len(synced) < 2:
            continue

        total_power = sum(p[3] for p in synced)
        defending   = garrison_at_arrival(tgt, anchor_eta, enemy_inc, friend_inc, my_player)
        needed      = ships_needed_to_capture(defending, False)

        if total_power < needed:
            continue

        # Assign ships proportionally
        total_sent  = 0
        for src, eta_p, angle_p, available_p in synced:
            if total_sent >= needed:
                break
            proportion = available_p / total_power
            my_share   = max(MIN_SEND, min(available_p, int(needed * proportion) + 2))
            if my_share < MIN_SEND or my_share > available_p:
                continue
            if not safe_to_attack(src, my_share, enemy_inc):
                continue

            # Precise ETA with actual share
            eta2, angle2 = find_intercept(
                src.x, src.y, tgt, my_share, ship_speed_max, omega
            )
            if eta2 is None:
                continue

            actions.append([src.id, float(angle2), int(my_share)])
            coord.register_action(tgt.id, my_share)
            committed_here.add(src.id)
            total_sent += my_share

    return actions

# ===========================================================================
# Winning-mode acceleration
# ===========================================================================

def plan_winning_push(my_planets, all_planets, enemy_inc, friend_inc,
                      coord, blocked_srcs, my_player, omega, ship_speed_max,
                      phase, prod_lead, deadline):
    """
    Aggressive endgame push: focus exclusively on eliminating enemy planets,
    send oversized fleets to overwhelm defences fast.
    """
    actions   = []
    used_srcs = set(blocked_srcs)

    enemy_targets = sorted(
        [p for p in all_planets if p.owner not in (-1, my_player)],
        key=lambda p: p.ships
    )
    if not enemy_targets:
        return [], used_srcs

    wp, we, wc, eb = phase_weights(phase, prod_lead)

    for src in sorted(
        [p for p in my_planets if p.id not in used_srcs],
        key=lambda p: -(p.ships - GARRISON_BASE)
    ):
        if time.time() >= deadline:
            break

        available = src.ships - GARRISON_BASE
        if available < MIN_SEND:
            continue

        best_move  = None
        best_score = -1e18

        for tgt in enemy_targets:
            if coord.is_covered(tgt, tgt.ships):
                continue

            eta, angle = find_intercept(src.x, src.y, tgt, available, ship_speed_max, omega)
            if eta is None:
                continue

            defending = garrison_at_arrival(tgt, eta, enemy_inc, friend_inc, my_player)
            already   = coord.committed(tgt.id)
            net_def   = max(0, defending - already)
            needed    = ships_needed_to_capture(net_def, False)
            send      = max(MIN_SEND, min(available, needed + CAPTURE_EXTRA_ENEMY))
            if send < MIN_SEND or send > available:
                continue
            if not safe_to_attack(src, send, enemy_inc):
                continue

            eta2, angle2 = find_intercept(src.x, src.y, tgt, send, ship_speed_max, omega)
            if eta2 is None:
                continue

            sc = target_score_with_denial(tgt, eta2, net_def, wp, we, wc, eb, prod_lead)
            if sc > best_score:
                best_score = sc
                best_move  = (tgt.id, float(angle2), int(send))

        if best_move:
            tid, angle, send = best_move
            actions.append([src.id, angle, send])
            coord.register_action(tid, send)
            used_srcs.add(src.id)

    return actions, used_srcs

# ===========================================================================
# Consolidation
# ===========================================================================

def plan_consolidation(my_planets, enemy_inc, friend_inc,
                       my_player, omega, ship_speed_max,
                       used_srcs, planet_roles, deadline):
    """Forward surplus rear ships to active front/mid planets."""
    actions = []

    rear_srcs = [
        p for p in my_planets
        if p.id not in used_srcs
        and planet_roles.get(p.id) == "rear"
        and enemy_inc.get(p.id, 0) == 0
        and (p.ships - GARRISON_BASE) >= CONSOLIDATE_THRESHOLD
    ]
    front_dests = [
        p for p in my_planets
        if planet_roles.get(p.id) in ("front", "mid")
        or p.ships < CONSOLIDATE_THRESHOLD
    ]

    if not rear_srcs or not front_dests:
        return []

    for src in rear_srcs:
        if time.time() >= deadline:
            break

        dest    = min(front_dests, key=lambda p: dist2(src.x, src.y, p.x, p.y))
        surplus = src.ships - GARRISON_BASE - CONSOLIDATE_THRESHOLD
        send    = min(surplus, src.ships - GARRISON_BASE)
        if send < MIN_SEND:
            continue

        eta, angle = find_intercept(src.x, src.y, dest, send, ship_speed_max, omega)
        if eta is None:
            continue

        actions.append([src.id, float(angle), int(send)])
        used_srcs.add(src.id)

    return actions

# ===========================================================================
# *** NEW *** Cascade threat response
#
# Uses ThreatSimulator to detect multi-wave attacks our reinforce planner
# would otherwise miss (it only sees current enemy_inc snapshot).
# Issues pre-emptive reinforcements from safe planets with the exact amount
# needed to survive all incoming waves.
# ===========================================================================

def plan_cascade_response(my_planets, planet_map, threat_sim,
                           enemy_inc, friend_inc, my_player,
                           omega, ship_speed_max,
                           blocked_srcs, coord, deadline):
    """
    For planets facing a wave cascade that immediate_threats() misses
    (second/third waves), issue reinforcements from safe donors.
    Returns (actions, used_srcs).
    """
    actions   = []
    used_srcs = set()

    for p in my_planets:
        if time.time() >= deadline:
            break
        if p.id in blocked_srcs:
            continue

        is_cascade, breach_turn = threat_sim.cascade_threat(p.id, p.ships)
        if not is_cascade or breach_turn is None:
            continue

        # Total incoming within breach_turn + 5 turns margin
        total_threat = threat_sim.total_incoming(p.id, breach_turn + 5)
        surplus_needed = max(0, total_threat - p.ships + GARRISON_BASE)
        if surplus_needed < MIN_SEND:
            continue

        # Find a safe donor
        donors = sorted(
            [q for q in my_planets
             if q.id != p.id
             and q.id not in blocked_srcs
             and q.id not in used_srcs
             and enemy_inc.get(q.id, 0) == 0
             and q.ships - GARRISON_BASE >= MIN_SEND],
            key=lambda q: dist2(q.x, q.y, p.x, p.y)
        )

        for donor in donors:
            available = donor.ships - GARRISON_BASE
            send      = min(available, surplus_needed + CAPTURE_EXTRA_ENEMY)
            send      = max(send, MIN_SEND)
            if send < MIN_SEND or send > available:
                continue

            eta, angle = find_intercept(
                donor.x, donor.y, p, send, ship_speed_max, omega
            )
            if eta is None:
                continue

            actions.append([donor.id, float(angle), int(send)])
            used_srcs.add(donor.id)
            coord.register_action(p.id, send)
            break

    return actions, used_srcs

# ===========================================================================
# *** NEW *** Orbital alignment double-capture
#
# When two neutral planets are in orbital alignment (passing near each other),
# a single fleet aimed at the intersection can potentially trigger capture of
# both (the engine captures the first planet the fleet hits).  We compute
# the alignment event and send a fleet sized to capture at least the first.
# This is a speculative opportunistic play — only fires when cheap.
# ===========================================================================

def plan_alignment_double_capture(my_planets, all_planets, enemy_inc, friend_inc,
                                   coord, blocked_srcs, my_player, omega,
                                   ship_speed_max, opa, deadline):
    """
    Opportunistic play: target pairs of orbiting neutrals that will briefly
    align (pass close to each other), sending one fleet to grab the first.
    Very low cost since the fleet was going there anyway.
    Returns (actions, used_srcs).
    """
    actions   = []
    used_srcs = set()

    neutrals = [
        p for p in all_planets
        if p.owner == -1
        and not coord.is_covered(p, p.ships)
        and _orbit_params(p)[0]
    ]
    if len(neutrals) < 2:
        return [], set()

    # Find alignment pairs
    alignment_pairs = []
    for i, pa in enumerate(neutrals):
        for pb in neutrals[i+1:]:
            t_align = opa.find_alignment_time(pa, pb, horizon=60)
            if t_align is not None:
                alignment_pairs.append((t_align, pa, pb))

    alignment_pairs.sort(key=lambda x: x[0])

    for t_align, pa, pb in alignment_pairs[:3]:  # limit to 3 pairs
        if time.time() >= deadline:
            break

        # Check if either target is already committed
        if coord.is_covered(pa, pa.ships) and coord.is_covered(pb, pb.ships):
            continue

        # Try to capture pa (the primary target)
        primary = pa if not coord.is_covered(pa, pa.ships) else pb

        for src in sorted(
            [p for p in my_planets if p.id not in blocked_srcs and p.id not in used_srcs],
            key=lambda p: dist2(p.x, p.y, primary.x, primary.y)
        ):
            if time.time() >= deadline:
                break
            available = src.ships - GARRISON_BASE
            if available < MIN_SEND:
                continue

            defending = garrison_at_arrival(primary, t_align, enemy_inc, friend_inc, my_player)
            needed    = ships_needed_to_capture(defending, True)
            send      = max(MIN_SEND, min(available, needed))
            if send > available:
                continue
            if not safe_to_attack(src, send, enemy_inc):
                continue

            eta, angle = find_intercept(
                src.x, src.y, primary, send, ship_speed_max, omega
            )
            if eta is None:
                continue

            actions.append([src.id, float(angle), int(send)])
            coord.register_action(primary.id, send)
            used_srcs.add(src.id)
            break

    return actions, used_srcs

# ===========================================================================
# Desperation: smart target selection using ThreatSimulator data
# ===========================================================================

def plan_smart_desperation(my_planets, all_planets, all_fleets,
                            omega, ship_speed_max, my_player,
                            enemy_inc, friend_inc, deadline):
    """
    Enhanced desperation: prioritise targets that can be reached before
    enemy fleets arrive to reinforce them, and send ships from every planet.
    """
    # Sort targets: first by ships asc (easiest), then by production desc (most value)
    targets = sorted(
        [p for p in all_planets if p.owner != my_player],
        key=lambda p: (p.ships, -p.production)
    )
    if not targets:
        return []

    actions   = []
    used_srcs = set()

    # Commit all planets — keep only 1 ship alive
    for src in sorted(my_planets, key=lambda p: -p.ships):
        if src.id in used_srcs:
            continue
        available = src.ships - 1
        if available < MIN_SEND:
            continue
        if time.time() >= deadline:
            break

        for tgt in targets:
            eta, angle = find_intercept(
                src.x, src.y, tgt, available, ship_speed_max, omega
            )
            if eta is None:
                continue
            actions.append([src.id, float(angle), int(available)])
            used_srcs.add(src.id)
            break

        if len(used_srcs) >= len(my_planets):
            break

    return actions

# ===========================================================================
# Main agent entry point
# ===========================================================================

# Module-level state for EnemyBehaviorModel (persists across turns)
_last_step = [-1]

def agent(obs, cfg=None):
    """
    NOVA v1 — Next-gen Orbital Victory Architecture.

    Called each turn by the Kaggle environment.
    obs  : Kaggle Struct or dict — current game observation
    cfg  : Kaggle Struct or dict — episode configuration (optional)

    Execution pipeline (priority order):
      0.  Parse + classify game state
      1.  Desperation — all-in if critically losing
      2.  Emergency evacuation — save ships from overwhelmed planets
      3.  Cascade threat response — multi-wave preemptive reinforcement
      4.  Counter-attack windows — hit weakened enemy planets
      5.  Multi-tier reinforcement (immediate + preemptive)
      6.  Snipe racing — reach neutrals before enemy fleets do
      7.  Alignment double-capture — opportunistic orbital pair captures
      8.  Early-game orbital rush (first RUSH_TURNS turns)
      9.  Winning-mode crush — aggressive elimination finish
     10.  Orbital-sync multi-assault — pooled timed coordinated strikes
     11.  Production-denial targeting — destroy top enemy producers first
     12.  Satellite-denial — preempt enemy forward staging planets
     13.  Global greedy attack — worldwide two-pass assignment
     14.  Adaptive fleet splitting — claim multiple cheap neutrals
     15.  Consolidation — forward surplus rear ships

    Final safety filter: drop any fleet with ships < MIN_SEND (=11).
    """
    t_start  = time.time()
    deadline = t_start + TIME_BUDGET

    # ------------------------------------------------------------ 0. Parse
    (step, my_player, omega, ship_speed_max,
     planets, fleets, comet_ids) = parse_obs(obs, cfg)

    max_steps  = 500   # standard Kaggle episode length
    planet_map = {p.id: p for p in planets}
    my_planets = [p for p in planets if p.owner == my_player]
    enemy_flts = [f for f in fleets if f.owner != my_player]

    if not my_planets:
        return []

    # ------------------------------------------------------------ Maps
    enemy_inc, friend_inc = build_incoming_maps(fleets, planets, my_player)

    # ------------------------------------------------------------ Update behavior model
    if step != _last_step[0]:
        update_enemy_model(fleets, planets, my_player)
        _last_step[0] = step

    # ------------------------------------------------------------ State classification
    phase          = game_phase(planets, my_player)
    prod_lead      = production_gap(planets, my_player)
    is_win         = winning_mode(planets, fleets, my_player)
    in_desp        = desperation_mode(planets, fleets, my_player)
    planet_roles   = classify_planets(my_planets, planets, my_player)
    enemy_planets  = [p for p in planets if p.owner not in (-1, my_player)]

    # ------------------------------------------------------------ Advanced systems
    opa = OrbitalPhaseAnalyzer(planets, omega, ship_speed_max)

    my_enemy_planets_for_map = enemy_planets if enemy_planets else [
        p for p in planets if p.owner not in (-1, my_player)
    ]
    influence_map = InfluenceMap(my_planets, my_enemy_planets_for_map,
                                  ship_speed_max)

    projector = EconomicProjector(planets, fleets, my_player, ship_speed_max)

    threat_sim = ThreatSimulator(
        enemy_flts, planets, my_player, ship_speed_max, omega
    )

    # ------------------------------------------------------------ Threat analysis
    imm_threats = immediate_threats(my_planets, enemy_inc, friend_inc)
    pre_threats = preemptive_threats(
        my_planets, enemy_planets, my_player, ship_speed_max
    )

    # ------------------------------------------------------------ 1. Desperation
    if in_desp:
        raw = plan_smart_desperation(
            my_planets, planets, fleets, omega, ship_speed_max,
            my_player, enemy_inc, friend_inc, deadline
        )
        return [a for a in raw if a[2] >= MIN_SEND]

    # ------------------------------------------------------------ Coordinator init
    coord = GlobalCoordinator()

    # ------------------------------------------------------------ 2. Emergency evacuation
    evac_blocked = set()
    evac_actions = plan_emergency_evacuation(
        my_planets, planet_map, imm_threats,
        enemy_inc, friend_inc, my_player,
        omega, ship_speed_max, evac_blocked, deadline
    )
    for act in evac_actions:
        coord.register_action(act[0], act[2])
    blocked_srcs = set(evac_blocked)

    # ------------------------------------------------------------ 3. Cascade response
    cascade_actions, cascade_srcs = plan_cascade_response(
        my_planets, planet_map, threat_sim,
        enemy_inc, friend_inc, my_player,
        omega, ship_speed_max,
        blocked_srcs, coord, deadline
    )
    blocked_srcs |= cascade_srcs

    # ------------------------------------------------------------ 4. Counter-attack
    counter_actions = find_counter_attack_windows(
        planets, fleets, my_player,
        my_planets, enemy_inc, friend_inc,
        ship_speed_max, omega, blocked_srcs, coord, deadline
    )
    for act in counter_actions:
        blocked_srcs.add(act[0])

    # ------------------------------------------------------------ 5. Reinforcements
    reinforce_actions, reinforce_srcs, reinforce_dests = plan_reinforcements(
        my_planets, planet_map, imm_threats, pre_threats,
        enemy_inc, friend_inc, my_player,
        omega, ship_speed_max, planet_roles, deadline
    )
    blocked_srcs |= reinforce_srcs
    for dest_id in reinforce_dests:
        coord.register_action(
            dest_id,
            imm_threats.get(dest_id, MIN_SEND) + CAPTURE_EXTRA_ENEMY
        )

    # ------------------------------------------------------------ 6. Snipe racing
    snipe_actions, snipe_srcs = plan_snipe_captures(
        my_planets, planets, enemy_flts,
        enemy_inc, friend_inc, my_player,
        omega, ship_speed_max, blocked_srcs,
        planet_map, coord, deadline
    )
    blocked_srcs |= snipe_srcs

    # ------------------------------------------------------------ 7. Alignment captures
    align_actions, align_srcs = [], set()
    if phase in ("early", "mid") and time.time() < deadline:
        align_actions, align_srcs = plan_alignment_double_capture(
            my_planets, planets, enemy_inc, friend_inc,
            coord, blocked_srcs, my_player, omega,
            ship_speed_max, opa, deadline
        )
        blocked_srcs |= align_srcs

    # ------------------------------------------------------------ 8. Early rush
    rush_actions, rush_srcs = [], set()
    if step < RUSH_TURNS and phase == "early" and time.time() < deadline:
        rush_actions, rush_srcs = plan_early_rush(
            my_planets, planets, enemy_inc, friend_inc,
            my_player, omega, ship_speed_max,
            blocked_srcs, coord, deadline, influence_map
        )
        blocked_srcs |= rush_srcs

    # ------------------------------------------------------------ 9. Winning push
    winning_actions, winning_srcs = [], set()
    if is_win and time.time() < deadline:
        winning_actions, winning_srcs = plan_winning_push(
            my_planets, planets, enemy_inc, friend_inc,
            coord, blocked_srcs, my_player, omega, ship_speed_max,
            phase, prod_lead, deadline
        )
        blocked_srcs |= winning_srcs

    # ------------------------------------------------------------ 10. Orbital sync assault
    sync_actions = []
    if (len(my_planets) >= 2 and not is_win
            and phase in ("mid", "late") and time.time() < deadline):
        sync_actions = plan_orbital_sync_assault(
            my_planets, planets, enemy_inc, friend_inc,
            coord, set(blocked_srcs), my_player,
            omega, ship_speed_max, phase, prod_lead,
            deadline, opa
        )
        for act in sync_actions:
            blocked_srcs.add(act[0])

    # ------------------------------------------------------------ 11. Production denial
    denial_actions, denial_srcs = [], set()
    if not is_win and phase in ("mid", "late") and time.time() < deadline:
        denial_actions, denial_srcs = plan_production_denial(
            my_planets, planets, enemy_inc, friend_inc,
            coord, blocked_srcs, my_player, omega,
            ship_speed_max, phase, prod_lead,
            projector, step, max_steps, deadline
        )
        blocked_srcs |= denial_srcs

    # ------------------------------------------------------------ 12. Satellite denial
    satellite_actions, satellite_srcs = [], set()
    if not is_win and enemy_planets and time.time() < deadline:
        satellite_actions, satellite_srcs = plan_satellite_denial(
            my_planets, planets, enemy_inc, friend_inc,
            coord, blocked_srcs, my_player, omega,
            ship_speed_max, deadline, influence_map
        )
        blocked_srcs |= satellite_srcs

    # ------------------------------------------------------------ 13. Global attacks
    attack_actions, attack_srcs = [], set()
    if not is_win and time.time() < deadline:
        attack_actions, attack_srcs = plan_global_attacks(
            my_planets, planets, enemy_inc, friend_inc,
            coord, blocked_srcs, my_player, omega,
            ship_speed_max, comet_ids, phase,
            prod_lead, False, projector, step, max_steps, deadline
        )

    # ------------------------------------------------------------ 14. Adaptive split
    split_actions, split_srcs = [], set()
    if phase in ("early", "mid") and time.time() < deadline:
        all_used_so_far = blocked_srcs | attack_srcs | winning_srcs
        split_actions, split_srcs = plan_adaptive_split(
            my_planets, planets, enemy_inc, friend_inc,
            coord, all_used_so_far, my_player, omega,
            ship_speed_max, phase, deadline
        )

    # ------------------------------------------------------------ 15. Consolidation
    all_used = (blocked_srcs | attack_srcs | winning_srcs | split_srcs)
    consolidate_actions = []
    if time.time() < deadline:
        consolidate_actions = plan_consolidation(
            my_planets, enemy_inc, friend_inc,
            my_player, omega, ship_speed_max,
            all_used, planet_roles, deadline
        )

    # ------------------------------------------------------------ Combine & filter
    all_actions = (
        evac_actions
        + cascade_actions
        + counter_actions
        + reinforce_actions
        + snipe_actions
        + align_actions
        + rush_actions
        + winning_actions
        + sync_actions
        + denial_actions
        + satellite_actions
        + attack_actions
        + split_actions
        + consolidate_actions
    )

    # Hard safety filter: MIN_SEND = 11 is the absolute floor.
    return [
        a for a in all_actions
        if isinstance(a, list) and len(a) == 3 and a[2] >= MIN_SEND
    ]
