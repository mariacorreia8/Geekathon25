import os
import pickle
import random
import time
from collections import defaultdict

try:
    import traci
except ImportError:
    raise RuntimeError("TraCI not found. Set PYTHONPATH to SUMO/tools.")

# --- SUMO Configuration ---
SUMO_BINARY = "sumo-gui"
SUMO_NET = "mapCruzamentoPequeno.net.xml"
SUMO_ROUTE = "routes2.rou.xml"
SUMO_ADDITIONAL = None

# --- Q-Learning Parameters (must match training script) ---
Q_TABLES_FILE = "q_table_multi_ep1998.pkl"
STATE_BINS = [0, 2, 5, 10, 15]  # Use the same bins as the training script
DECISION_INTERVAL = 10          # Use the same interval as the training script

# ----------------- Helpers -----------------
def q_table_factory():
    return defaultdict(list)

def start_sumo():
    cmd = [SUMO_BINARY, "-n", SUMO_NET, "-r", SUMO_ROUTE, "--start", "--step-length", "0.5"]
    if SUMO_ADDITIONAL:
        cmd += ["-a", SUMO_ADDITIONAL]
    traci.start(cmd)


def discretize(count):
    for i, threshold in enumerate(STATE_BINS):
        if count <= threshold:
            return i
    return len(STATE_BINS)

def get_local_state(tls_id):
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    unique_lanes = sorted(list(set(lanes)))
    return tuple(discretize(traci.lane.getLastStepHaltingNumber(l)) for l in unique_lanes)

def choose_action(q_table, state, num_actions):
    """
    Chooses the best action (greedy) based on the Q-table for a given state.
    """
    if state not in q_table:
        # If state is unknown, default to a random action or phase 0
        return random.randrange(num_actions)
    
    q_row = q_table[state]
    max_q = max(q_row)
    # Return the index of the first occurrence of the max value
    return q_row.index(max_q)

# ----------------- Run Q-learning Control -----------------
def run():
    # Load the dictionary of Q-tables from the file
    try:
        with open(Q_TABLES_FILE, "rb") as f:
            raw_data = pickle.load(f)
            q_tables = defaultdict(q_table_factory, raw_data)
        print("Loaded Q-tables from file.")
    except (FileNotFoundError, AttributeError) as e:
        print(f"Error loading Q-tables: {e}")
        print("Please ensure the training script has been run and the file exists and is not corrupted.")
        return

    start_sumo()
    tls_list = traci.trafficlight.getIDList()
    if not tls_list:
        print("No traffic lights found.")
        traci.close()
        return

    # Initialize states and actions for each agent
    agents = {}
    for tls_id in tls_list:
        agents[tls_id] = {
            'num_actions': len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases),
            'state': get_local_state(tls_id),
        }
    
    steps = 0
    MAX_STEPS = 3600

    while steps < MAX_STEPS:
        # --- For each agent: choose and apply an action ---
        for tls_id, agent in agents.items():
            # Choose the best action based on the agent's current state and its Q-table
            action = choose_action(q_tables[tls_id], agent['state'], agent['num_actions'])
            
            # Apply action if it's a green phase
            # This is important to allow yellow/red transitions to complete
            traci.trafficlight.setPhase(tls_id, action)

        # --- Simulate for DECISION_INTERVAL steps ---
        for _ in range(DECISION_INTERVAL):
            traci.simulationStep()
            time.sleep(0.1)  # slow down for visualization
            steps += 1
            if steps >= MAX_STEPS:
                break
        
        # --- Update states for the next decision cycle ---
        for tls_id, agent in agents.items():
            agent['state'] = get_local_state(tls_id)

    traci.close()
    print("Simulation finished.")

if __name__ == "__main__":
    random.seed(0)
    run()