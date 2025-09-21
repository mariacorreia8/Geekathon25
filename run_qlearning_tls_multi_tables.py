import os
import pickle
import random
import time
import csv
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
    cmd = [SUMO_BINARY, "-n", SUMO_NET, "-r", SUMO_ROUTE, "--step-length", "0.1"]
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

    # ----------------- NEW: stopped-time tracking setup -----------------
    stopped_times = {}   # {vehicleID: accumulated_stopped_time}
    step_length = traci.simulation.getDeltaT()
    controlled_lanes = set(l for tls in tls_list for l in traci.trafficlight.getControlledLanes(tls))

    steps = 0
    MAX_STEPS = 3600

    while steps < MAX_STEPS:
        # --- For each agent: choose and apply an action ---
        for tls_id, agent in agents.items():
            action = choose_action(q_tables[tls_id], agent['state'], agent['num_actions'])
            traci.trafficlight.setPhase(tls_id, action)

        # --- Simulate for DECISION_INTERVAL steps ---
        for _ in range(DECISION_INTERVAL):
            traci.simulationStep()
            time.sleep(0.1)  # slow down for visualization
            steps += 1
            if steps >= MAX_STEPS:
                break

            # ----------------- NEW: stopped-time tracking -----------------
            for veh_id in traci.vehicle.getIDList():
                speed = traci.vehicle.getSpeed(veh_id)
                lane_id = traci.vehicle.getLaneID(veh_id)

                # only count if the vehicle is in a traffic-light-controlled lane
                if lane_id in controlled_lanes and speed < 0.5:
                    if veh_id not in stopped_times:
                        stopped_times[veh_id] = 0.0
                    stopped_times[veh_id] += step_length

        # --- Update states for the next decision cycle ---
        for tls_id, agent in agents.items():
            agent['state'] = get_local_state(tls_id)

    traci.close()
    print("Simulation finished.")

    # ----------------- NEW: Save results -----------------
    print("Stopped times at traffic lights (optimized run):")
    for veh_id, time_s in stopped_times.items():
        print(f"{veh_id}: {time_s:.2f} s")

    with open("stopped_times_optimized.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["VehicleID", "StoppedTime(s)"])
        for veh_id, time_s in stopped_times.items():
            writer.writerow([veh_id, f"{time_s:.2f}"])

if __name__ == "__main__":
    random.seed(0)
    run()
