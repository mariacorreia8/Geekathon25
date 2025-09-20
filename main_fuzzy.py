import os
import random
import numpy as np
from collections import defaultdict

# Using scikit-fuzzy for the fuzzy logic controller.
# You'll need to install it: pip install scikit-fuzzy
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except ImportError:
    raise RuntimeError("scikit-fuzzy not found. Please install it with 'pip install scikit-fuzzy'")

# ==== Config ====
SUMO_BINARY = "sumo"  # or "sumo-gui"
SUMO_NET = "mapCruzamentoPequeno.net.xml"
SUMO_ROUTE = "routes2.rou.xml"
SUMO_ADDITIONAL = None

DECISION_INTERVAL = 5
MIN_GREEN = 10  # minimum green time for a phase (seconds)
# =================

try:
    import traci
except ImportError:
    raise RuntimeError("TraCI not found. Add SUMO/tools to PYTHONPATH.")


# ---- Helpers ----
def start_sumo():
    """Starts the SUMO simulation."""
    cmd = [SUMO_BINARY, "-n", SUMO_NET, "-r", SUMO_ROUTE, "--start"]
    if SUMO_ADDITIONAL:
        cmd += ["-a", SUMO_ADDITIONAL]
    traci.start(cmd)


def group_lanes(lanes):
    """Group lanes by approximate direction: north/south/east/west"""
    groups = defaultdict(list)
    for l in lanes:
        if l.startswith('-') or '0' in l:  # rough grouping; adjust based on net
            groups['N'].append(l)
        elif '1' in l:
            groups['S'].append(l)
        elif '2' in l:
            groups['E'].append(l)
        else:
            groups['W'].append(l)
    return groups


# ---- Fuzzy Logic Controller ----
def create_fuzzy_controller():
    """Creates the fuzzy logic controller for a single intersection."""
    # Define fuzzy variables for input and output
    queue_length = ctrl.Antecedent(np.arange(0, 11, 1), 'queue_length')
    phase_change = ctrl.Consequent(np.arange(0, 11, 1), 'phase_change')

    # Define membership functions for inputs
    queue_length['low'] = fuzz.trimf(queue_length.universe, [0, 0, 3])
    queue_length['medium'] = fuzz.trimf(queue_length.universe, [1, 5, 9])
    queue_length['high'] = fuzz.trimf(queue_length.universe, [7, 10, 10])

    # Define membership functions for outputs
    phase_change['no_change'] = fuzz.trimf(phase_change.universe, [0, 0, 5])
    phase_change['change'] = fuzz.trimf(phase_change.universe, [5, 10, 10])

    # Define fuzzy rules
    rule1 = ctrl.Rule(queue_length['low'], phase_change['no_change'])
    rule2 = ctrl.Rule(queue_length['medium'], phase_change['change'])
    rule3 = ctrl.Rule(queue_length['high'], phase_change['change'])

    # Create the control system
    fuzzy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    return ctrl.ControlSystemSimulation(fuzzy_ctrl)


# ---- Simulation Loop ----
def simulate_with_fuzzy_logic():
    """Runs the SUMO simulation with a fuzzy logic controller."""
    start_sumo()
    tls_ids = traci.trafficlight.getIDList()
    tls_info = {}

    for tls in tls_ids:
        lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls)))
        groups = group_lanes(lanes)
        n_actions = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0].phases)
        
        # Determine which phases correspond to which lane groups
        phase_map = {}
        ryg_def = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0]
        for i, phase in enumerate(ryg_def.phases):
            green_lanes = [l for l in lanes if 'g' in phase.state[traci.trafficlight.getLinkIndex(tls, l)]]
            if any(l in groups['N'] + groups['S'] for l in green_lanes):
                phase_map['NS'] = i
            elif any(l in groups['E'] + groups['W'] for l in green_lanes):
                phase_map['EW'] = i

        tls_info[tls] = dict(
            groups=groups,
            n_actions=n_actions,
            phase=traci.trafficlight.getPhase(tls),
            last_change=0,
            controller=create_fuzzy_controller(),
            phase_map=phase_map
        )

    steps = 0
    while steps < 3600:  # Use a fixed simulation duration
        # Choose actions based on fuzzy logic
        for tls, info in tls_info.items():
            # Enforce minimum green time
            if steps - info["last_change"] < MIN_GREEN:
                action = info["phase"]
            else:
                # Calculate queue lengths for each group
                queue_ns = sum(traci.lane.getLastStepHaltingNumber(l) for l in info['groups']['N'] + info['groups']['S'])
                queue_ew = sum(traci.lane.getLastStepHaltingNumber(l) for l in info['groups']['E'] + info['groups']['W'])

                # Determine which queue is longer
                current_phase = info["phase"]
                if current_phase == info['phase_map']['NS']:
                    # Currently serving N/S traffic, consider E/W queue
                    input_queue = queue_ew
                    next_phase = info['phase_map']['EW']
                else:
                    # Currently serving E/W traffic, consider N/S queue
                    input_queue = queue_ns
                    next_phase = info['phase_map']['NS']
                
                # Use fuzzy controller to decide
                info['controller'].input['queue_length'] = min(input_queue, 10)  # Clamp to max fuzzy input
                info['controller'].compute()
                
                phase_change_value = info['controller'].output['phase_change']
                
                if phase_change_value > 5:
                    action = next_phase
                    if action != current_phase:
                        info["last_change"] = steps
                else:
                    action = current_phase

            traci.trafficlight.setPhase(tls, action)
            info.update(action=action, phase=traci.trafficlight.getPhase(tls))

        # Simulate DECISION_INTERVAL steps
        for _ in range(DECISION_INTERVAL):
            traci.simulationStep()
            steps += 1
            if steps >= 3600:
                break

    traci.close()
    print("Simulation finished.")

if __name__ == "__main__":
    random.seed(0)
    try:
        simulate_with_fuzzy_logic()
    except KeyboardInterrupt:
        print("Interrupted.")
        try:
            traci.close()
        except:
            pass