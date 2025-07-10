# Configuration file for the Digital Twin Simulation

# Simulation Parameters
SIMULATION_TIME_HOURS = 24 * 7  # Simulate for one week

# Manufacturing Facility Parameters
FACILITY_LAYOUT = {
    'size_x': 500,  # meters
    'size_y': 500,  # meters
    'locations': {
        'machine_1': (50, 50),
        'machine_2': (50, 450),
        'machine_3': (450, 450),
        'machine_4': (450, 50),
        'warehouse': (250, 250),
        'charging_station': (0, 250),
    }
}

# Vehicle (AGV) Parameters
NUM_VEHICLES = 10
VEHICLE_SPECS = {
    'battery_capacity_kwh': 5.0,
    'charging_rate_kw': 3.0,
    'energy_consumption_kwh_per_km': 0.2,
    'speed_kmh': 10,
    'min_soc_percent': 20, # Minimum allowable state of charge
}

# Charging Station Parameters
NUM_CHARGING_PORTS = 4

# Workload Parameters
TASK_ARRIVAL_RATE_PER_HOUR = 5 # Average number of new tasks per hour

# Electricity Price Data (Placeholder - to be replaced with real data)
# Using a simple time-of-use structure for now
# Prices are in $/kWh
ELECTRICITY_PRICES_HOURLY = [
    (0, 8, 0.3784),
    (8, 11, 0.9014),
    (11, 13, 0.3784),
    (13, 19, 0.9014),
    (19, 21, 1.2064),
    (21, 22, 0.9014),
    (22, 24, 0.3784),
]

# Optimization Parameters
OPTIMIZATION_HORIZON_MINUTES = 60 # Look ahead 60 minutes
TIME_STEP_MINUTES = 10 # Re-evaluate the schedule every 10 minutes

# Costs
VEHICLE_DOWNTIME_COST_PER_HOUR = 100 # Arbitrary cost for an idle vehicle
