"""
qlearning_tls_light.py

Lightweight tabular Q-learning controller for a single SUMO traffic light.
"""

import os, sys, pickle, random
from collections import defaultdict

SUMO_BINARY = "sumo"  # use sumo-gui if you want GUI
SUMO_NET = "mapCruzamentoPequeno.net.xml"
SUMO_ROUTE = "routes2.rou.xml"
SUMO_ADDITIONAL = None

# Q-learning params
ALPHA = 0.7
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
NUM_EPISODES = 200
MAX_STEPS = 3600
DECISION_INTERVAL = 5
STATE_BINS = [0, 1, 3]
Q_TABLE_FILE = "q_table.pkl"
PHASE_CHANGE_PENALTY = 0.1

try:
    import traci
except ImportError:
    raise RuntimeError("TraCI not found. Set PYTHONPATH to SUMO/tools.")

# ----------------- Helpers -----------------
def start_sumo():
    cmd = [SUMO_BINARY, "-n", SUMO_NET, "-r", SUMO_ROUTE, "--start"]
    if SUMO_ADDITIONAL:
        cmd += ["-a", SUMO_ADDITIONAL]
    traci.start(cmd)

def discretize(count):
    for i, t in enumerate(STATE_BINS):
        if count <= t: return i
    return len(STATE_BINS)

def get_state(lanes):
    return tuple(discretize(traci.lane.getLastStepHaltingNumber(l)) for l in lanes)

def state_key(state):
    return ",".join(map(str, state))

def choose_action(q_table, key, n_actions, epsilon):
    if random.random() < epsilon or key not in q_table:
        return random.randrange(n_actions)
    q_row = q_table[key]
    max_q = max(q_row)
    return random.choice([i for i, q in enumerate(q_row) if q == max_q])

def ensure_q_row(q_table, key, n_actions):
    if key not in q_table:
        q_table[key] = [0.0] * n_actions

# ----------------- Training -----------------
def train():
    q_table = {}
    if os.path.exists(Q_TABLE_FILE):
        with open(Q_TABLE_FILE, "rb") as f:
            q_table = pickle.load(f)
        print("Loaded Q-table.")

    epsilon = EPSILON_START

    for ep in range(1, NUM_EPISODES + 1):
        start_sumo()
        tls_list = traci.trafficlight.getIDList()
        if not tls_list:
            print("No traffic lights found.")
            traci.close()
            return
        tls_id = tls_list[0]
        lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
        n_actions = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases)

        traci.simulationStep()
        prev_state = get_state(lanes)
        prev_key = state_key(prev_state)
        ensure_q_row(q_table, prev_key, n_actions)
        prev_halts = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)

        total_reward = 0
        steps = 0
        current_phase = traci.trafficlight.getPhase(tls_id)

        while steps < MAX_STEPS:
            action = choose_action(q_table, prev_key, n_actions, epsilon)
            phase_changed = int(action != current_phase)
            traci.trafficlight.setPhase(tls_id, action)
            current_phase = action

            for _ in range(DECISION_INTERVAL):
                traci.simulationStep()
                steps += 1
                if steps >= MAX_STEPS:
                    break

            cur_state = get_state(lanes)
            cur_key = state_key(cur_state)
            ensure_q_row(q_table, cur_key, n_actions)

            cur_halts = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
            reward = (prev_halts - cur_halts) - PHASE_CHANGE_PENALTY * phase_changed

            # Q-learning update
            q_table[prev_key][action] += ALPHA * (
                reward + GAMMA * max(q_table[cur_key]) - q_table[prev_key][action]
            )
            total_reward += reward

            prev_key = cur_key
            prev_halts = cur_halts

        traci.close()
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Save checkpoint every 3 episodes
        if ep % 3 == 0:
            checkpoint_file = f"q_table_ep{ep}.pkl"
            with open(checkpoint_file, "wb") as f:
                pickle.dump(q_table, f)
            print(f"Checkpoint saved: {checkpoint_file}")

        print(f"Episode {ep} done. Reward: {total_reward:.2f}, epsilon: {epsilon:.3f}")

    # Save final Q-table
    with open(Q_TABLE_FILE, "wb") as f:
        pickle.dump(q_table, f)
    print("Training finished. Final Q-table saved.")



if __name__ == "__main__":
    random.seed(0)
    try:
        train()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        try: traci.close()
        except: pass
