"""
qlearning_tls_multi_robust_v2.py

Robust multi-TLS tabular Q-learning for SUMO with:
- Lane grouping (N/S/E/W)
- Proper average waiting time tracking
- Normalized reward
- Minimum green time enforcement
- Multi-TLS support (separate Q-tables)
"""

import os, pickle, random
from collections import defaultdict

# ==== Config ====
SUMO_BINARY = "sumo"  # or "sumo-gui"
SUMO_NET = "mapCruzamentoPequeno.net.xml"
SUMO_ROUTE = "routes2.rou.xml"
SUMO_ADDITIONAL = None

ALPHA, GAMMA = 0.1, 0.95
EPSILON_START, EPSILON_END, EPSILON_DECAY = 1.0, 0.05, 0.999
NUM_EPISODES, MAX_STEPS, DECISION_INTERVAL = 1000, 3600, 5
STATE_BINS = [0, 1, 3]  # 0,1,2-3,4+
PHASE_CHANGE_PENALTY = 0.1
MIN_GREEN = 10  # minimum green time for a phase (seconds)
SAVE_INTERVAL = 3
Q_TABLE_FILE = "q_table_multi_robust.pkl"
# =================

try:
    import traci
except ImportError:
    raise RuntimeError("TraCI not found. Add SUMO/tools to PYTHONPATH.")

# ---- Helpers ----
def start_sumo():
    cmd = [SUMO_BINARY, "-n", SUMO_NET, "-r", SUMO_ROUTE, "--start"]
    if SUMO_ADDITIONAL:
        cmd += ["-a", SUMO_ADDITIONAL]
    traci.start(cmd)

def discretize(x):
    for i, t in enumerate(STATE_BINS):
        if x <= t: return i
    return len(STATE_BINS)

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

def get_state(groups, phase):
    """State: current phase + discretized sum of halts per direction"""
    bins = [discretize(sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)) 
            for lanes in groups.values()]
    return (phase,) + tuple(bins)

def state_key(s):
    return ",".join(map(str, s))

def choose_action(q_table, key, n, eps):
    if random.random() < eps or key not in q_table:
        return random.randrange(n)
    row = q_table[key]
    m = max(row)
    return random.choice([i for i, q in enumerate(row) if q == m])

def ensure_q(q_table, key, n):
    if key not in q_table:
        q_table[key] = [0.0] * n

# ---- Training ----
def train():
    q_tables = {}
    if os.path.exists(Q_TABLE_FILE):
        with open(Q_TABLE_FILE, "rb") as f:
            q_tables = pickle.load(f)
        print("Loaded Q-tables.")

    eps = EPSILON_START

    for ep in range(1, NUM_EPISODES + 1):
        start_sumo()
        tls_ids = traci.trafficlight.getIDList()
        tls_info = {}
        
        for tls in tls_ids:
            lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls)))
            groups = group_lanes(lanes)
            n_actions = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0].phases)
            if tls not in q_tables:
                q_tables[tls] = {}
            phase = traci.trafficlight.getPhase(tls)
            s = get_state(groups, phase)
            key = state_key(s)
            ensure_q(q_tables[tls], key, n_actions)
            tls_info[tls] = dict(groups=groups, n_actions=n_actions,
                                 state_key=key, phase=phase,
                                 last_change=0)

        total_reward, steps = 0, 0
        all_veh_ids = set()
        total_wait = 0.0

        while steps < MAX_STEPS:
            # choose actions
            for tls, info in tls_info.items():
                # enforce minimum green
                if steps - info["last_change"] < MIN_GREEN:
                    action = info["phase"]
                    changed = 0
                else:
                    action = choose_action(q_tables[tls], info["state_key"], info["n_actions"], eps)
                    changed = int(action != info["phase"])
                    if changed: info["last_change"] = steps
                traci.trafficlight.setPhase(tls, action)
                info.update(action=action, phase_changed=changed)

            # simulate DECISION_INTERVAL steps
            for _ in range(DECISION_INTERVAL):
                traci.simulationStep()
                steps += 1
                if steps >= MAX_STEPS: break
                veh_ids = traci.vehicle.getIDList()
                all_veh_ids.update(veh_ids)
                total_wait += sum(traci.vehicle.getWaitingTime(v) for v in veh_ids)

            # update Q-tables
            for tls, info in tls_info.items():
                phase = traci.trafficlight.getPhase(tls)
                s = get_state(info["groups"], phase)
                key = state_key(s)
                ensure_q(q_tables[tls], key, info["n_actions"])
                halts = sum(sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes) 
                            for lanes in info["groups"].values())
                reward = -(halts / max(1,len(info["groups"]))) - PHASE_CHANGE_PENALTY * info["phase_changed"]
                
                old_q = q_tables[tls][info["state_key"]][info["action"]]
                q_tables[tls][info["state_key"]][info["action"]] = old_q + ALPHA * (
                    reward + GAMMA * max(q_tables[tls][key]) - old_q
                )
                total_reward += reward
                info.update(state_key=key, phase=phase)

        traci.close()
        eps = max(EPSILON_END, eps * EPSILON_DECAY)
        avg_wait = total_wait / len(all_veh_ids) if all_veh_ids else 0.0

        # save every SAVE_INTERVAL episodes
        if ep % SAVE_INTERVAL == 0:
            ckpt = f"q_table_multi_ep{ep}.pkl"
            with open(ckpt, "wb") as f: pickle.dump(q_tables, f)
            print(f"Checkpoint saved: {ckpt}")

        print(f"Episode {ep}/{NUM_EPISODES} done. Total reward {total_reward:.2f}, Avg wait {avg_wait:.2f}s, eps {eps:.3f}")

    # save final Q-tables
    with open(Q_TABLE_FILE, "wb") as f: pickle.dump(q_tables, f)
    print("Training finished. Final Q-tables saved.")

if __name__ == "__main__":
    random.seed(0)
    try:
        train()
    except KeyboardInterrupt:
        print("Interrupted.")
        try: traci.close()
        except: pass
