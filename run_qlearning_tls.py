import os, pickle, random, time 
import numpy as np
try:
    import traci
except ImportError:
    raise RuntimeError("TraCI not found. Set PYTHONPATH to SUMO/tools.")

SUMO_BINARY = "sumo-gui"  # or "sumo-gui" for visualization
SUMO_NET = "mapCruzamentoPequeno.net.xml"
SUMO_ROUTE = "routes2.rou.xml"
SUMO_ADDITIONAL = None

STATE_BINS = [0, 1, 3]
DECISION_INTERVAL = 5
Q_TABLE_FILE = "q_table_ep3.pkl"
PHASE_CHANGE_PENALTY = 0.1

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

def choose_action(q_table, key, n_actions):
    if key not in q_table:
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

# ----------------- Run Q-learning Control -----------------
def run():
    with open(Q_TABLE_FILE, "rb") as f:
        q_table = pickle.load(f)
    print("Loaded Q-table. Running SUMO simulation...")

    start_sumo()

    # Step once so TLS info is available
    traci.simulationStep()

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

    # estado inicial das fases
    current_actions = [traci.trafficlight.getPhase(tls_id) for tls_id in tls_list]

    steps = 0
    MAX_STEPS = 3600  # or whatever sim length you want

    while steps < MAX_STEPS:
        # always exploit (greedy)
        action_idx = int(np.argmax(q_table[prev_key]))
        action_tuple = joint_actions[action_idx]

        # apply new phases if needed
        for tls_id, phase in zip(tls_list, action_tuple):
            traci.trafficlight.setPhase(tls_id, phase)
        current_actions = list(action_tuple)

        # run simulation for DECISION_INTERVAL steps
        for _ in range(DECISION_INTERVAL):
            traci.simulationStep()
            time.sleep(0.1)  # slow down if you want to watch, remove otherwise
            steps += 1
            if steps >= MAX_STEPS:
                break

        # update state for next decision
        cur_state = get_global_state(tls_list)
        prev_key = state_key(cur_state)
        ensure_q_row(q_table, prev_key, len(joint_actions))

    traci.close()
    print("Simulation finished.")


if __name__ == "__main__":
    random.seed(0)
    run()
