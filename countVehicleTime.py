import traci
import sumolib

# Start SUMO (change sumo-gui to sumo if you don’t need the GUI)
sumoBinary = "sumo-gui"
sumoCmd = [sumoBinary, "-c", "mapCruzamentoPequeno.sumocfg"]

traci.start(sumoCmd)

stopped_times = {}   # {vehicleID: accumulated_stopped_time}
stopped_flags = {}   # {vehicleID: whether currently stopped}

step_length = traci.simulation.getDeltaT() / 1000.0  # step size in seconds

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    for veh_id in traci.vehicle.getIDList():
        speed = traci.vehicle.getSpeed(veh_id)

        # consider stopped if speed < 0.1 m/s
        if speed < 0.1:
            if veh_id not in stopped_times:
                stopped_times[veh_id] = 0.0
                stopped_flags[veh_id] = True
            elif stopped_flags[veh_id]:
                stopped_times[veh_id] += step_length
        else:
            stopped_flags[veh_id] = False

    # Optional: print live updates
    # print(stopped_times)

traci.close()

# Print final results
print("Stopped times at traffic lights:")
for veh_id, time in stopped_times.items():
    print(f"{veh_id}: {time:.2f} s")
