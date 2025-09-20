import os, pickle, random, time 
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
Q_TABLE_FILE = "q_table_multi_robust.pkl"
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

    tls_id = tls_list[0]
    lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))

    programs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
    if not programs or not programs[0].phases:
        print("Traffic light program not available yet.")
        traci.close()
        return

    n_actions = len(programs[0].phases)

    current_phase = traci.trafficlight.getPhase(tls_id)
    prev_state = get_state(lanes)
    prev_key = state_key(prev_state)

    steps = 0
    MAX_STEPS = 3600  # total simulation steps

    while steps < MAX_STEPS:
        action = choose_action(q_table, prev_key, n_actions)
        phase_changed = int(action != current_phase)
        traci.trafficlight.setPhase(tls_id, action)
        current_phase = action

        for _ in range(DECISION_INTERVAL):
            traci.simulationStep()
            time.sleep(0.4)
            steps += 1
            if steps >= MAX_STEPS:
                break

        prev_state = get_state(lanes)
        prev_key = state_key(prev_state)

    traci.close()
    print("Simulation finished.")

if __name__ == "__main__":
    random.seed(0)
    run()
