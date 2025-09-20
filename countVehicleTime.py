import traci
import sumolib
import csv


# Start SUMO (change sumo-gui to sumo if you don’t need the GUI)
sumoBinary = "sumo-gui"
sumoCmd = [sumoBinary, "-c", "mapCruzamentoPequeno.sumocfg"]

traci.start(sumoCmd)

stopped_times = {}   # {vehicleID: accumulated_stopped_time}
stopped_flags = {}   # {vehicleID: whether currently stopped}

step_length = traci.simulation.getDeltaT() 

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    for veh_id in traci.vehicle.getIDList():
        speed = traci.vehicle.getSpeed(veh_id)

        # consider stopped if speed < 0.1 m/s
        if speed < 0.5:
            if veh_id not in stopped_times:
                stopped_times[veh_id] = 0.0
            stopped_times[veh_id] += step_length

    # Optional: print live updates
    # print(stopped_times)

traci.close()


# Print final results
print("Stopped times at traffic lights:")
for veh_id, time in stopped_times.items():
    print(f"{veh_id}: {time:.2f} s")


#save to CSV 
with open("stopped_times.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["VehicleID", "StoppedTime(s)"])  # header
    for veh_id, time in stopped_times.items():
        writer.writerow([veh_id, f"{time:.2f}"])

