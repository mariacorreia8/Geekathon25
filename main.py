"""
qlearning_tls.py

Minimal tabular Q-learning controller for a single SUMO traffic light (intersection).
- Works with existing phases (so respects turn restrictions already present in your net/connection XML).
- State: discretized halting vehicle counts per incoming lane.
- Action: choose a phase index (0..phaseCount-1).
- Reward: reduction in halting vehicles (so it learns to reduce stops).

Requirements:
    - SUMO (with TraCI python available)
    - Python 3.8+
    - pickle (standard)

Adjust parameters below as needed.
"""

import os
import sys
import time
import pickle
import random
from collections import defaultdict

# ========== USER: set these paths / binary ==========
SUMO_BINARY = "sumo-gui"   # or "sumo" if you don’t want the GUI
SUMO_NET = "mapCruzamentoPequeno.net.xml"
SUMO_ROUTE = "routes2.rou.xml"
SUMO_ADDITIONAL = None #"tls.add.xml"  # if your tlLogic is stored here
# ===================================================

# Q-learning params
ALPHA = 0.7            # learning rate
GAMMA = 0.9            # discount factor
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
NUM_EPISODES = 200     # # of training episodes
MAX_STEPS = 3600       # timesteps per episode (simulation seconds)
DECISION_INTERVAL = 5  # seconds between decisions (agent picks new phase every 5s)
STATE_BINS = [0, 1, 3] # discretization thresholds for halting vehicles: 0, 1, 2-3, 4+
Q_TABLE_FILE = "q_table.pkl"

# reward shaping
PHASE_CHANGE_PENALTY = 0.1

# TraCI import
try:
    import traci
except Exception as e:
    print("Failed to import traci. Make sure SUMO's tools are in PYTHONPATH.")
    print("If SUMO is installed, e.g.: export PYTHONPATH=$SUMO_HOME/tools")
    raise

# ----------------- Helpers -----------------
def start_sumo():
    sumo_cmd = [SUMO_BINARY, "-n", SUMO_NET, "-r", SUMO_ROUTE, "--start"]
    if SUMO_ADDITIONAL:
        sumo_cmd += ["-a", SUMO_ADDITIONAL]
    # no gui by default; if you want gui use "sumo-gui" as SUMO_BINARY
    traci.start(sumo_cmd)

def discretize_count(count):
    """Map integer count -> discrete bin index (0..len(STATE_BINS))."""
    for i, thresh in enumerate(STATE_BINS):
        if count <= thresh:
            return i
    return len(STATE_BINS)

def make_state_from_lane_counts(lane_ids):
    """Return a tuple of discretized halting counts for the provided lanes (order preserved)."""
    return tuple(discretize_count(int(traci.lane.getLastStepHaltingNumber(l))) for l in lane_ids)

def state_to_key(state):
    """Convert state tuple to string key for dict indexing (stable)."""
    return ",".join(map(str, state))

def choose_action(q_table, state_key, n_actions, epsilon):
    """Epsilon-greedy action selection."""
    if random.random() < epsilon:
        return random.randrange(n_actions)
    # choose best action(s)
    q_row = q_table.get(state_key)
    if q_row is None:
        return random.randrange(n_actions)
    max_q = max(q_row)
    # tie-breaker random among best
    best_actions = [i for i, q in enumerate(q_row) if q == max_q]
    return random.choice(best_actions)

def ensure_q_row(q_table, state_key, n_actions):
    if state_key not in q_table:
        q_table[state_key] = [0.0] * n_actions

# --------------- Main training loop ---------------
def train():
    # load or init Q-table
    if os.path.exists(Q_TABLE_FILE):
        with open(Q_TABLE_FILE, "rb") as f:
            q_table = pickle.load(f)
        print("Loaded Q-table from", Q_TABLE_FILE)
    else:
        q_table = dict()

    epsilon = EPSILON_START

    for episode in range(1, NUM_EPISODES + 1):
        start_sumo()
        tls_ids = traci.trafficlight.getIDList()
        if len(tls_ids) == 0:
            print("No traffic lights found in the network. Exiting.")
            traci.close()
            return
        tls_id = tls_ids[0]  # choose the first TLS found; change if you want a specific tls
        print(f"Episode {episode}/{NUM_EPISODES} — controlling TLS: {tls_id}")

        # get controlled lanes (incoming lanes). This respects your net connections / restrictions.
        # getControlledLanes returns lanes that are in the TLS program (may include outgoing lanes).
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        # Keep only unique lanes and prefer incoming (lane IDs that start with '-') or simply trust this list.
        # We'll keep the order stable:
        lane_ids = []
        for l in controlled_lanes:
            if l not in lane_ids:
                lane_ids.append(l)

        n_actions = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases)
        print(f"Detected {len(lane_ids)} controlled lanes, {n_actions} phases.")

        # initialize state
        traci.simulationStep()  # advance one step to initialize lane data
        prev_state = make_state_from_lane_counts(lane_ids)
        prev_state_key = state_to_key(prev_state)
        ensure_q_row(q_table, prev_state_key, n_actions)

        # record previous halts for reward calc
        prev_total_halts = sum(int(traci.lane.getLastStepHaltingNumber(l)) for l in lane_ids)

        total_reward_episode = 0.0
        steps = 0
        current_phase = traci.trafficlight.getPhase(tls_id)

        while steps < MAX_STEPS:
            # choose action every DECISION_INTERVAL seconds
            action = choose_action(q_table, prev_state_key, n_actions, epsilon)
            # apply action (phase index)
            if action != current_phase:
                phase_changed = 1
            else:
                phase_changed = 0
            traci.trafficlight.setPhase(tls_id, action)
            current_phase = action

            # simulate DECISION_INTERVAL seconds
            for _ in range(DECISION_INTERVAL):
                traci.simulationStep()
                steps += 1
                if steps >= MAX_STEPS:
                    break

            # observe new state and reward
            cur_state = make_state_from_lane_counts(lane_ids)
            cur_state_key = state_to_key(cur_state)
            ensure_q_row(q_table, cur_state_key, n_actions)

            cur_total_halts = sum(int(traci.lane.getLastStepHaltingNumber(l)) for l in lane_ids)
            # reward = reduction in halting vehicles (positive if halts decreased)
            reward = (prev_total_halts - cur_total_halts) - PHASE_CHANGE_PENALTY * phase_changed

            # Q-learning update
            old_q = q_table[prev_state_key][action]
            max_next_q = max(q_table[cur_state_key])
            new_q = old_q + ALPHA * (reward + GAMMA * max_next_q - old_q)
            q_table[prev_state_key][action] = new_q

            total_reward_episode += reward

            # advance state
            prev_state_key = cur_state_key
            prev_total_halts = cur_total_halts

            # (optional) small early-stopping if traffic became zero
            if cur_total_halts == 0 and steps > 20:
                # no halting vehicles left — good episode
                pass

            # print occasional debug
            if steps % (DECISION_INTERVAL * 20) == 0:
                print(f" episode {episode} step {steps}/{MAX_STEPS} reward_so_far {total_reward_episode:.2f} eps {epsilon:.3f}")

            if steps >= MAX_STEPS:
                break

        # end episode
        traci.close()
        # decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # save Q-table periodically
        with open(Q_TABLE_FILE, "wb") as f:
            pickle.dump(q_table, f)

        print(f"Episode {episode} ended. Total reward: {total_reward_episode:.2f}, epsilon: {epsilon:.3f}")

    print("Training finished. Q-table saved to", Q_TABLE_FILE)

# --------------- Run as script ---------------
if __name__ == "__main__":
    random.seed(0)
    try:
        train()
    except KeyboardInterrupt:
        print("User interrupted. Closing TraCI if open.")
        try:
            traci.close()
        except:
            pass
