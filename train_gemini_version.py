"""
qlearning_tls_light_improved.py

Multi-agent (Independent Learners) tabular Q-learning controller for SUMO traffic lights.
Each traffic light is an independent agent with its own Q-table, state, and reward.
"""

import os
import sys
import pickle
import random
from collections import defaultdict

# --- SUMO Configuration ---
SUMO_BINARY = "sumo"  # Use "sumo-gui" for headless mode
SUMO_NET = "mapCruzamentoPequeno.net.xml"
SUMO_ROUTE = "routes2.rou.xml"
SUMO_ADDITIONAL = None

# --- Q-Learning Hyperparameters ---
ALPHA = 0.2                 # Learning rate: Lower for more stable learning
GAMMA = 0.95                # Discount factor: High to value future rewards
EPSILON_START = 1.0         # Exploration rate start
EPSILON_END = 0.05          # Exploration rate end
EPSILON_DECAY = 0.999       # Decay rate: slightly faster for simpler problem
NUM_EPISODES = 2000         # Can be reduced, as each agent's problem is simpler
MAX_STEPS = 3600            # Max steps per episode
DECISION_INTERVAL = 10      # Agent makes a decision every N seconds
PHASE_CHANGE_PENALTY = 0.1  # Reduced penalty, as reward signal is stronger
COLLISION_PENALTY = 50      # Large negative reward for collisions

# --- State/Action Definition ---
STATE_BINS = [0, 2, 5, 10, 15] # More granular bins for better state definition
Q_TABLES_FILE = "q_table_multi_robust_have_collision.pkl"

try:
    import traci
except ImportError:
    raise RuntimeError("TraCI not found. Set PYTHONPATH to SUMO/tools.")

# ----------------- Helper Functions -----------------

def q_table_factory():
    return defaultdict(list)

def start_sumo():
    """Starts a new SUMO simulation instance."""
    cmd = [SUMO_BINARY, "-n", SUMO_NET, "-r", SUMO_ROUTE, "--start", "--quit-on-end"]
    if SUMO_ADDITIONAL:
        cmd += ["-a", SUMO_ADDITIONAL]
    traci.start(cmd)

def discretize(count):
    """Discretizes a continuous car count into a bin index."""
    for i, threshold in enumerate(STATE_BINS):
        if count <= threshold:
            return i
    return len(STATE_BINS)

def get_local_state(tls_id):
    """Gets the state for a single traffic light agent."""
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    # Use a sorted list of unique lanes to ensure consistent state representation
    unique_lanes = sorted(list(set(lanes)))
    return tuple(discretize(traci.lane.getLastStepHaltingNumber(l)) for l in unique_lanes)

def choose_action(q_table, state, num_actions, epsilon):
    """Chooses an action using an epsilon-greedy policy."""
    if random.random() < epsilon or state not in q_table:
        return random.randrange(num_actions)
    
    q_row = q_table[state]
    max_q = max(q_row)
    # If multiple actions have the same max Q-value, choose one randomly
    return random.choice([i for i, q in enumerate(q_row) if q == max_q])

def get_local_reward(tls_id, prev_waits):
    """
    Calculates the reward for a single agent.
    Reward is the negative change in total waiting time on its controlled lanes.
    A positive reward means waiting time has decreased.
    """
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    unique_lanes = sorted(list(set(lanes)))
    
    current_waits = {l: traci.lane.getWaitingTime(l) for l in unique_lanes}
    reward = sum(prev_waits.get(l, 0) - current_waits.get(l, 0) for l in unique_lanes)

    # Check for collisions in the simulation
    collision_count = traci.simulation.getCollidingVehiclesNumber() 
    if collision_count > 0:
        # Apply a large negative reward for each crash
        reward -= COLLISION_PENALTY * collision_count
    return reward, current_waits

# ----------------- Main Training Loop -----------------

def train():
    # q_tables is a dictionary: {tls_id_1: q_table_1, tls_id_2: q_table_2, ...}
    q_tables = defaultdict(q_table_factory)
    if os.path.exists(Q_TABLES_FILE):
        with open(Q_TABLES_FILE, "rb") as f:
            q_tables = pickle.load(f)
        print("Loaded Q-tables from file.")

    epsilon = EPSILON_START

    for ep in range(1, NUM_EPISODES + 1):
        start_sumo()
        
        tls_list = traci.trafficlight.getIDList()
        if not tls_list:
            print("Error: No traffic lights found in the network.")
            traci.close()
            return

        # Initialize agents
        agents = {}
        for tls_id in tls_list:
            num_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases)
            agents[tls_id] = {
                'num_actions': num_phases,
                'state': get_local_state(tls_id),
                'action': 0,
                'reward': 0,
                'total_reward': 0,
                'prev_waits': {l: 0 for l in traci.trafficlight.getControlledLanes(tls_id)}
            }
            # Ensure Q-table rows exist for initial states
            if agents[tls_id]['state'] not in q_tables[tls_id]:
                q_tables[tls_id][agents[tls_id]['state']] = [0.0] * num_phases
        
        step = 0
        while step < MAX_STEPS:
            # --- For each agent: choose and apply an action ---
            for tls_id, agent in agents.items():
                prev_state = agent['state']
                
                # Choose action
                action = choose_action(
                    q_tables[tls_id], prev_state, agent['num_actions'], epsilon
                )
                
                # Apply action if it's a green phase
                current_phase = traci.trafficlight.getPhase(tls_id)
                # Only apply action on green phases to avoid controlling yellow/red transitions
                if action != current_phase and action % 2 == 0: 
                    traci.trafficlight.setPhase(tls_id, action)

                agent['action'] = action
                agent['prev_state'] = prev_state

            # --- Simulate for DECISION_INTERVAL steps ---
            for _ in range(DECISION_INTERVAL):
                if step >= MAX_STEPS: break
                traci.simulationStep()
                step += 1

                # Check for collisions after each step
                if traci.simulation.getCollidingVehiclesNumber() > 0:
                    # A crash happened. Penalize all agents for this action.
                    for tls_id, agent in agents.items():
                        reward = -COLLISION_PENALTY
                        # Apply the penalty to the Q-table update
                        prev_q = q_tables[tls_id][agent['prev_state']][agent['action']]
                        new_q = prev_q + ALPHA * (reward + GAMMA * 0 - prev_q) # No next_max_q as episode might end
                        q_tables[tls_id][agent['prev_state']][agent['action']] = new_q
                    traci.close()
                    break # End the episode immediately

            # --- For each agent: calculate reward and update Q-table ---
            for tls_id, agent in agents.items():
                # Get new state and reward
                current_state = get_local_state(tls_id)
                reward, new_waits = get_local_reward(tls_id, agent['prev_waits'])
                agent['prev_waits'] = new_waits

                # Penalize for changing phase
                if agent['action'] != traci.trafficlight.getPhase(tls_id):
                    reward -= PHASE_CHANGE_PENALTY
                
                # Ensure Q-table row exists for the new state
                if current_state not in q_tables[tls_id]:
                    q_tables[tls_id][current_state] = [0.0] * agent['num_actions']

                # Q-learning update formula
                prev_q = q_tables[tls_id][agent['prev_state']][agent['action']]
                next_max_q = max(q_tables[tls_id][current_state])
                
                new_q = prev_q + ALPHA * (reward + GAMMA * next_max_q - prev_q)
                q_tables[tls_id][agent['prev_state']][agent['action']] = new_q
                
                agent['state'] = current_state
                agent['total_reward'] += reward

        traci.close()
        
        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        # Log results
        total_reward_ep = sum(a['total_reward'] for a in agents.values())
        print(f"Episode {ep}/{NUM_EPISODES} | Total Reward: {total_reward_ep:.2f} | Epsilon: {epsilon:.4f}")

        # Save Q-tables periodically
        if ep % 10 == 0:
            with open(Q_TABLES_FILE, "wb") as f:
                pickle.dump(q_tables, f)

    print("\nTraining finished.")
    with open(Q_TABLES_FILE, "wb") as f:
        pickle.dump(q_tables, f)
    print(f"Final Q-tables saved to {Q_TABLES_FILE}")

if __name__ == "__main__":
    try:
        train()
    except traci.exceptions.TraCIException as e:
        print(f"Error connecting to SUMO: {e}")
        print("Please ensure SUMO is installed and the network/route files are correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        try:
            traci.close()
        except:
            pass