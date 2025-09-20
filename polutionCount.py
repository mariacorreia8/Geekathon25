import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter.font as tkFont

# Load CSV
df = pd.read_csv("vehicles_with_numbers.csv")

# Emission ranges
gas_emissions = {"CO2": (3, 7), "CO": (0.006, 0.014), "NOx": (0.0003, 0.014)}
diesel_emissions = {"CO2": 0.2, "CO": 0.001, "NOx": 0.003}
electric_emissions = {"CO2": 0, "CO": 0, "NOx": 0}

# Calculate emissions
def calculate_emissions(row):
    t = row['StoppedTime(s)']
    v = row['NewNumber']
    if v == 2:  # Gas
        co2 = np.random.uniform(*gas_emissions["CO2"]) * t
        co = np.random.uniform(*gas_emissions["CO"]) * t
        nox = np.random.uniform(*gas_emissions["NOx"]) * t
    elif v == 1:  # Diesel
        co2 = diesel_emissions["CO2"] * t
        co = diesel_emissions["CO"] * t
        nox = diesel_emissions["NOx"] * t
    else:  # Electric
        co2 = co = nox = 0
    return pd.Series([co2, co, nox])

# apply emissions calculations (adds three columns to df)
df[['CO2(g)', 'CO(g)', 'NOx(g)']] = df.apply(calculate_emissions, axis=1)

# friendly names for vehicle types (1,2,3)
vehicle_names = {1: "Diesel", 2: "Gasoline", 3: "Electric"}

# aggregate totals per vehicle type
summary = df.groupby('NewNumber')[['CO2(g)', 'CO(g)', 'NOx(g)']].sum()
# map index numbers to friendly names (safer with .get)
summary.index = [vehicle_names.get(int(i), str(i)) for i in summary.index]

# ----------------- Tkinter UI -----------------
root = tk.Tk()
root.title("Vehicle Emissions Dashboard")
root.geometry("1100x750")  # larger window

# Fonts (bigger, as requested)
header_font = tkFont.Font(family="Helvetica", size=14, weight="bold")
label_font = tkFont.Font(family="Helvetica", size=13)
tree_font = tkFont.Font(family="Helvetica", size=12)

# ----------------- Top frame: Sections 1 & 2 side by side -----------------
top_frame = tk.Frame(root)
top_frame.pack(fill="x", padx=15, pady=15)

# Section 1: Pollution by Vehicle Type
frame1 = tk.LabelFrame(top_frame, text="Pollution by Vehicle Type", padx=15, pady=15)
frame1.pack(side="left", fill="both", expand=True, padx=10)

# Internal frame for table and chart side by side
pollution_frame = tk.Frame(frame1)
pollution_frame.pack(fill="both", expand=True)

# Table (left)
tree = ttk.Treeview(pollution_frame)
tree["columns"] = ("Vehicle", "CO2(g)", "CO(g)", "NOx(g)")
tree.heading("#0", text="", anchor="center")
tree.column("#0", width=0, stretch=tk.NO)
for col in tree["columns"]:
    tree.heading(col, text=col, anchor="center")
    tree.column(col, width=180, anchor="center")

# Style table rows
style = ttk.Style()
style.configure("Treeview", rowheight=40, font=tree_font)
style.configure("Treeview.Heading", font=header_font)

# Insert aggregated totals into table
for v_type, row in summary.iterrows():
    tree.insert("", tk.END, values=(v_type,
                                    round(row['CO2(g)'], 2),
                                    round(row['CO(g)'], 4),
                                    round(row['NOx(g)'], 4)))
tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)

# Bar Chart (right)
fig, ax = plt.subplots(figsize=(6,5))
summary.plot(kind='bar', ax=ax)
ax.set_ylabel("Emissions (grams)", fontsize=13)
ax.set_title("Total Emissions per Vehicle Type", fontsize=14)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
plt.tight_layout()

canvas = FigureCanvasTkAgg(fig, master=pollution_frame)
canvas.draw()
canvas.get_tk_widget().pack(side="left", fill="both", expand=True, padx=10, pady=10)

# Section 2: Simulation Info (right of Section 1)
frame2 = tk.LabelFrame(top_frame, text="Simulation Info", padx=15, pady=15)
frame2.pack(side="left", fill="both", expand=True, padx=10)

# --- THE KEY METRICS ---
total_vehicles = len(df)

# Total stopped time = sum of the second field (StoppedTime(s)), in seconds -> convert to minutes
total_stopped_seconds = df['StoppedTime(s)'].sum()
total_stopped_minutes = total_stopped_seconds / 60.0

# Simulation duration (elapsed time) - here shown as the longest single stopped time (in minutes)
simulation_duration_seconds = df['StoppedTime(s)'].max()
simulation_duration_minutes = simulation_duration_seconds / 60.0

# Average stop time per vehicle (in minutes)
average_stop_minutes = df['StoppedTime(s)'].mean() / 60.0

# Display metrics
tk.Label(frame2, text=f"Total Vehicles: {total_vehicles}", font=label_font).pack(anchor="w", pady=10)
tk.Label(frame2, text=f"Total Stopped Time (all cars): {total_stopped_minutes:.2f} minutes", font=label_font).pack(anchor="w", pady=8)
tk.Label(frame2, text=f"Average Stop Time per Vehicle: {average_stop_minutes:.2f} minutes", font=label_font).pack(anchor="w", pady=8)

# ----------------- Section 3: Vehicle Lookup -----------------
frame3 = tk.LabelFrame(root, text="Vehicle Lookup", padx=15, pady=15)
frame3.pack(fill="x", padx=15, pady=15)

tk.Label(frame3, text="Select Vehicle ID:", font=label_font).pack(side="left", padx=5)

vehicle_var = tk.IntVar()
vehicle_choices = df['VehicleID'].tolist()
dropdown = ttk.Combobox(frame3, textvariable=vehicle_var, values=vehicle_choices, font=label_font, width=12)
dropdown.pack(side="left", padx=5)

result_label = tk.Label(frame3, text="", font=label_font)
result_label.pack(side="left", padx=20)

def show_vehicle_emissions(event=None):
    vid = vehicle_var.get()
    vehicle_row = df[df['VehicleID'] == vid]
    if not vehicle_row.empty:
        co2 = vehicle_row['CO2(g)'].values[0]
        co = vehicle_row['CO(g)'].values[0]
        nox = vehicle_row['NOx(g)'].values[0]
        stopped_s = vehicle_row['StoppedTime(s)'].values[0]
        stopped_min = stopped_s / 60.0
        result_label.config(text=f"Stopped: {stopped_min:.2f} min — CO2: {co2:.2f} g, CO: {co:.4f} g, NOx: {nox:.4f} g")
    else:
        result_label.config(text="Vehicle not found")

dropdown.bind("<<ComboboxSelected>>", show_vehicle_emissions)

root.mainloop()
