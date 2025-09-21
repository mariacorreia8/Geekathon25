import traci
import sumolib
import csv
import os
import random

# ----------------- SUMO Setup -----------------
sumoBinary = "sumo-gui"  # or "sumo" for no GUI
sumoCmd = [sumoBinary, "-c", "mapCruzamentoPequeno.sumocfg"]

traci.start(sumoCmd)

stopped_times = {}   # {vehicleID: accumulated_stopped_time}

step_length = traci.simulation.getDeltaT() 

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    for veh_id in traci.vehicle.getIDList():
        speed = traci.vehicle.getSpeed(veh_id)

        # consider stopped if speed < 0.5 m/s
        if speed < 0.5:
            if veh_id not in stopped_times:
                stopped_times[veh_id] = 0.0
            stopped_times[veh_id] += step_length

traci.close()

# ----------------- Save to CSV -----------------
csv_file = "vehicles_with_numbers_optimized.csv"

vehicle_ids = list(stopped_times.keys())
num_vehicles = len(vehicle_ids)

# Define vehicle type distribution
percentages = {1: 0.55, 2: 0.36, 3: 0.09}

# Calculate counts for each type
type_counts = {k: int(percentages[k]*num_vehicles) for k in percentages}

# Adjust rounding errors
while sum(type_counts.values()) < num_vehicles:
    type_counts[1] += 1  # add extra to Diesel if needed

# Create list with proper number of each type
vehicle_type_list = []
for v_type, count in type_counts.items():
    vehicle_type_list.extend([v_type]*count)

# Shuffle to randomly assign types to vehicle IDs
random.shuffle(vehicle_type_list)

# Write CSV
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["VehicleID", "StoppedTime(s)", "NewNumber"])  # header
    for veh_id, new_number in zip(vehicle_ids, vehicle_type_list):
        time = stopped_times[veh_id]
        writer.writerow([veh_id, f"{time:.2f}", new_number])

print(f"Saved stopped times and vehicle types to {csv_file}")
