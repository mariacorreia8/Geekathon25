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



STATE_BINS = [0, 1, 3]   # manter coarse bins
ALPHA = 0.2                 # Learning rate: Lower for more stable learning
GAMMA = 0.95                # Discount factor: High to value future rewards
EPSILON_START = 1.0         # Exploration rate start
EPSILON_END = 0.05          # Exploration rate end
EPSILON_DECAY = 0.999       # Decay rate: slightly faster for simpler problem
NUM_EPISODES = 2000         # Can be reduced, as each agent's problem is simpler
MAX_STEPS = 3600            # Max steps per episode
DECISION_INTERVAL = 10      # Agent makes a decision every N seconds
PHASE_CHANGE_PENALTY = 0.1  # Reduced penalty, as reward signal is stronger
COLLISION_PENALTY = 100

Q_TABLE_FILE = "q_table_global_agent_with_collision.pkl"
SAVE_INTERVAL = 5

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

def get_global_state(tls_list):
    state = []
    for tls_id in tls_list:
        lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
        for l in lanes:
            count = traci.lane.getLastStepHaltingNumber(l)
            state.append(discretize(count))
    return tuple(state)

def get_action_space(tls_list):
    return [len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases)
            for tls_id in tls_list]

def all_joint_actions(action_sizes):
    """Lista todas as combinações possíveis de fases para todos os TLS."""
    import itertools
    return list(itertools.product(*[range(n) for n in action_sizes]))

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

        action_sizes = get_action_space(tls_list)
        joint_actions = all_joint_actions(action_sizes)

        traci.simulationStep()
        prev_state = get_global_state(tls_list)
        prev_key = state_key(prev_state)
        ensure_q_row(q_table, prev_key, len(joint_actions))

        prev_halts = sum(traci.lane.getLastStepHaltingNumber(l)
                         for tls_id in tls_list
                         for l in traci.trafficlight.getControlledLanes(tls_id))

        total_reward = 0
        steps = 0
        # estado inicial das fases
        current_actions = [traci.trafficlight.getPhase(tls_id) for tls_id in tls_list]

        while steps < MAX_STEPS:
            action_idx = choose_action(q_table, prev_key, len(joint_actions), epsilon)
            action_tuple = joint_actions[action_idx]

            # aplicar fases em todos os TLS
            phase_changed = sum(int(action_tuple[i] != current_actions[i]) for i in range(len(tls_list)))
            for tls_id, phase in zip(tls_list, action_tuple):
                traci.trafficlight.setPhase(tls_id, phase)
            current_actions = list(action_tuple)

            for _ in range(DECISION_INTERVAL):
                traci.simulationStep()
                steps += 1
                if steps >= MAX_STEPS:
                    break

            # Check for collisions after each simulation step
            if traci.simulation.getCollidingVehiclesNumber() > 0:
                print(f"Collision detected! Applying large penalty.")
                reward = -COLLISION_PENALTY
                # Update Q-table with the large negative reward
                prev_q = q_table[prev_key][action_idx]
                new_q = prev_q + ALPHA * (reward + GAMMA * 0 - prev_q)
                q_table[prev_key][action_idx] = new_q
                break # End the episode immediately
                
            cur_state = get_global_state(tls_list)
            cur_key = state_key(cur_state)
            ensure_q_row(q_table, cur_key, len(joint_actions))

            cur_halts = sum(traci.lane.getLastStepHaltingNumber(l)
                            for tls_id in tls_list
                            for l in traci.trafficlight.getControlledLanes(tls_id))

            reward = (prev_halts - cur_halts) - PHASE_CHANGE_PENALTY * phase_changed

            # Q-learning update
            q_table[prev_key][action_idx] += ALPHA * (
                reward + GAMMA * max(q_table[cur_key]) - q_table[prev_key][action_idx]
            )
            total_reward += reward

            prev_key = cur_key
            prev_halts = cur_halts

        traci.close()
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        with open(Q_TABLE_FILE, "wb") as f:
            pickle.dump(q_table, f)
        print(f"Episode {ep}/{NUM_EPISODES} done. Reward: {total_reward:.2f}, epsilon: {epsilon:.3f}")

    print("Training finished. Q-table saved.")


if __name__ == "__main__":
    random.seed(0)
    try:
        train()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        try: traci.close()
        except: pass
