import sys
import pandas as pd
import numpy as np
import os

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QComboBox, QGroupBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


# ----------------- Check / Create CSV -----------------
csv_file = "vehicles_with_numbers_optimized.csv"

if not os.path.exists(csv_file):
    print(f"{csv_file} not found. Creating a default dataset...")
    # Create sample dataset
    num_vehicles = 20  # adjust as needed
    df = pd.DataFrame({
        "VehicleID": range(1, num_vehicles+1),
        "StoppedTime(s)": np.random.uniform(10, 300, size=num_vehicles),   # 10s to 5min
        "NewNumber": np.random.choice([1,2,3], size=num_vehicles),  # Diesel, Gas, Electric
    })
    df.to_csv(csv_file, index=False)
else:
    df = pd.read_csv(csv_file)

# ----------------- Emissions -----------------
gas_emissions = {"CO2": (3, 7), "CO": (0.006, 0.014), "NOx": (0.0003, 0.014)}
diesel_emissions = {"CO2": 0.2, "CO": 0.001, "NOx": 0.003}
electric_emissions = {"CO2": 0, "CO": 0, "NOx": 0}

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
    else:
        co2 = co = nox = 0
    return pd.Series([co2, co, nox])

df[['CO2(g)', 'CO(g)', 'NOx(g)']] = df.apply(calculate_emissions, axis=1)

# Vehicle names
vehicle_names = {1: "Diesel", 2: "Gasoline", 3: "Electric"}
summary = df.groupby('NewNumber')[['CO2(g)', 'CO(g)', 'NOx(g)']].sum()
summary.index = [vehicle_names.get(int(i), str(i)) for i in summary.index]


# ----------------- PyQt5 App -----------------
class Dashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vehicle Emissions Dashboard")
        self.setGeometry(100, 100, 1200, 800)

        # ----------------- DARK MODE -----------------
        dark_stylesheet = """
        QWidget {
            background-color: #121212;
            color: #e0e0e0;
            font-family: Helvetica;
        }
        QGroupBox {
            border: 1px solid #444444;
            margin-top: 10px;
            font-weight: bold;
            font-size: 13pt;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 5px 10px;
            color: #ffffff;
        }
        QTableWidget {
            background-color: #1e1e1e;
            gridline-color: #444444;
            alternate-background-color: #2e2e2e;
        }
        QHeaderView::section {
            background-color: #2e2e2e;
            color: #ffffff;
            font-weight: bold;
            font-size: 11pt;
        }
        QComboBox, QLabel {
            color: #e0e0e0;
        }
        QComboBox {
            background-color: #1e1e1e;
            selection-background-color: #3a3a3a;
        }
        QPushButton {
            background-color: #2e2e2e;
            color: #e0e0e0;
            border: 1px solid #555555;
            padding: 5px;
        }
        QPushButton:hover {
            background-color: #3a3a3a;
        }
        """
        self.setStyleSheet(dark_stylesheet)

        main_layout = QVBoxLayout()

        # Top layout: Table + Chart
        top_layout = QHBoxLayout()

        # ----- Table -----
        table_group = QGroupBox("Pollution by Vehicle Type")
        table_layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Vehicle", "CO2(g)", "CO(g)", "NOx(g)", "Total"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setFont(QFont("Helvetica", 11))
        self.load_table()
        table_layout.addWidget(self.table)
        table_group.setLayout(table_layout)
        top_layout.addWidget(table_group, 2)

        # ----- Chart -----
        chart_group = QGroupBox("Total Emissions per Vehicle Type")
        chart_layout = QVBoxLayout()
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(6,4), facecolor='#121212')
        ax.set_facecolor('#121212')
        summary.plot(kind='bar', ax=ax, color=['#1f77b4','#ff7f0e','#2ca02c'])
        ax.set_ylabel("Emissions (grams)", fontsize=12, color="#e0e0e0")
        ax.set_xlabel("Vehicle Type", fontsize=12, color="#e0e0e0")
        ax.tick_params(axis='x', labelsize=11, colors="#e0e0e0")
        ax.tick_params(axis='y', labelsize=11, colors="#e0e0e0")
        for spine in ax.spines.values():
            spine.set_color('#444444')
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        chart_layout.addWidget(canvas)
        chart_group.setLayout(chart_layout)
        top_layout.addWidget(chart_group, 3)

        main_layout.addLayout(top_layout)

        # ----- Metrics -----
        metrics_group = QGroupBox("Simulation Info")
        metrics_layout = QVBoxLayout()
        total_vehicles = len(df)
        total_stopped_min = df['StoppedTime(s)'].sum()/60
        avg_stop_min = df['StoppedTime(s)'].mean()/60
        metrics = [
            f"Total Vehicles: {total_vehicles}",
            f"Total Stopped Time (all cars): {total_stopped_min:.2f} minutes",
            f"Average Stop Time per Vehicle: {avg_stop_min:.2f} minutes"
        ]
        for m in metrics:
            lbl = QLabel(m)
            lbl.setFont(QFont("Helvetica", 12))
            metrics_layout.addWidget(lbl)
        metrics_group.setLayout(metrics_layout)
        main_layout.addWidget(metrics_group)

        # ----- Vehicle Lookup -----
        lookup_group = QGroupBox("Vehicle Lookup")
        lookup_layout = QHBoxLayout()
        lookup_label = QLabel("Select Vehicle ID:")
        lookup_label.setFont(QFont("Helvetica", 12))
        self.vehicle_dropdown = QComboBox()
        self.vehicle_dropdown.addItems([str(i) for i in df['VehicleID'].tolist()])
        self.vehicle_dropdown.setFont(QFont("Helvetica", 12))
        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Helvetica", 12))
        self.vehicle_dropdown.currentIndexChanged.connect(self.show_vehicle_emissions)
        lookup_layout.addWidget(lookup_label)
        lookup_layout.addWidget(self.vehicle_dropdown)
        lookup_layout.addWidget(self.result_label)
        lookup_group.setLayout(lookup_layout)
        main_layout.addWidget(lookup_group)

        self.setLayout(main_layout)

    def load_table(self):
        self.table.setRowCount(len(summary))
        for row_idx, (v_type, row) in enumerate(summary.iterrows()):
            total = row['CO2(g)'] + row['CO(g)'] + row['NOx(g)']
            self.table.setItem(row_idx, 0, QTableWidgetItem(v_type))
            self.table.setItem(row_idx, 1, QTableWidgetItem(f"{row['CO2(g)']:.2f}"))
            self.table.setItem(row_idx, 2, QTableWidgetItem(f"{row['CO(g)']:.4f}"))
            self.table.setItem(row_idx, 3, QTableWidgetItem(f"{row['NOx(g)']:.4f}"))
            self.table.setItem(row_idx, 4, QTableWidgetItem(f"{total:.4f}"))
        self.table.resizeColumnsToContents()

    def show_vehicle_emissions(self):
        vid = int(self.vehicle_dropdown.currentText())
        vehicle_row = df[df['VehicleID'] == vid]
        if not vehicle_row.empty:
            co2 = vehicle_row['CO2(g)'].values[0]
            co = vehicle_row['CO(g)'].values[0]
            nox = vehicle_row['NOx(g)'].values[0]
            stopped_min = vehicle_row['StoppedTime(s)'].values[0]/60
            self.result_label.setText(
                f"Stopped: {stopped_min:.2f} min — CO2: {co2:.2f} g, CO: {co:.4f} g, NOx: {nox:.4f} g"
            )
        else:
            self.result_label.setText("Vehicle not found")


# ----------------- Run App -----------------
app = QApplication(sys.argv)
window = Dashboard()
window.show()
sys.exit(app.exec_())
