import simpy
import random
import numpy as np
import pandas as pd
from . import config
from . import optimization_controller

class ChargingDataProcessor:
    def __init__(self, charging_data_path, weather_data_path):
        self.charging_data = pd.read_csv(charging_data_path, encoding='latin1')
        self.weather_data = pd.read_csv(weather_data_path, encoding='latin1')
        self._preprocess_data()

    def _preprocess_data(self):
        # Convert time columns to datetime objects
        self.charging_data['Start Time'] = pd.to_datetime(self.charging_data['Start Time'])
        self.charging_data['End Time'] = pd.to_datetime(self.charging_data['End Time'])
        self.charging_data['Order creation time'] = pd.to_datetime(self.charging_data['Order creation time'])
        self.charging_data['Payment time'] = pd.to_datetime(self.charging_data['Payment time'])

        self.weather_data['Date'] = pd.to_datetime(self.weather_data['Date'])

        # Handle missing values (example: fill with mean or median, or drop rows)
        self.charging_data.fillna(0, inplace=True) # Example: fill NaN with 0
        self.weather_data.fillna(method='ffill', inplace=True) # Example: forward fill NaN

        # Calculate charging duration and power
        self.charging_data['ChargingDuration_minutes'] = (self.charging_data['End Time'] - self.charging_data['Start Time']).dt.total_seconds() / 60
        self.charging_data['ChargingPower_kW'] = self.charging_data['Transaction power/kwh'] / (self.charging_data['ChargingDuration_minutes'] / 60)

    def get_random_charging_event(self):
        # Sample a random charging event
        return self.charging_data.sample(1).iloc[0]

    def get_weather_data_for_time(self, timestamp):
        # Get weather data for a specific timestamp (e.g., nearest hour)
        date = timestamp.floor('H')
        weather = self.weather_data[self.weather_data['Date'] == date]
        if not weather.empty:
            return weather.iloc[0]
        return None

class DataCollector:
    """A class to collect and report simulation data."""
    def __init__(self):
        self.total_energy_consumed = 0
        self.tasks_completed = 0
        self.tasks_generated = 0
        self.vehicle_downtime = {i: 0 for i in range(config.NUM_VEHICLES)}
        self.last_downtime_start = {i: 0 for i in range(config.NUM_VEHICLES)}

    def record_charge(self, energy):
        self.total_energy_consumed += energy

    def record_task_completion(self):
        self.tasks_completed += 1

    def record_task_generation(self):
        self.tasks_generated += 1

    def start_downtime(self, vehicle_id, time):
        self.last_downtime_start[vehicle_id] = time

    def end_downtime(self, vehicle_id, time):
        downtime = time - self.last_downtime_start[vehicle_id]
        self.vehicle_downtime[vehicle_id] += downtime

    def report(self):
        print("\n--- Simulation Report ---")
        print(f"Total Energy Consumed: {self.total_energy_consumed:.2f} kWh")
        print(f"Total Tasks Generated: {self.tasks_generated}")
        print(f"Total Tasks Completed: {self.tasks_completed}")
        total_downtime = sum(self.vehicle_downtime.values())
        print(f"Total Vehicle Downtime: {total_downtime / 3600:.2f} hours")
        print("-------------------------\n")

class Vehicle:
    """Models an Electric Vehicle (AGV) in the simulation."""
    def __init__(self, env, vehicle_id, task_queue, charging_station, data_collector, strategy='uncontrolled'):
        self.env = env
        self.vehicle_id = vehicle_id
        self.task_queue = task_queue
        self.charging_station = charging_station
        self.data_collector = data_collector
        self.strategy = strategy
        self.soc = config.VEHICLE_SPECS['battery_capacity_kwh']
        self.location = config.FACILITY_LAYOUT['locations']['charging_station']
        self.action = env.process(self.run())
        self.charging_schedule = []

    def run(self):
        """Main process loop for the vehicle."""
        if self.strategy == 'uncontrolled':
            yield self.env.process(self.run_uncontrolled())
        elif self.strategy == 'fcfs':
            yield self.env.process(self.run_fcfs())
        elif self.strategy == 'intelligent':
            yield self.env.process(self.run_intelligent())

    def run_uncontrolled(self):
        """Implements the Uncontrolled Charging (charge-when-low) baseline."""
        while True:
            min_soc = config.VEHICLE_SPECS['min_soc_percent'] / 100.0 * config.VEHICLE_SPECS['battery_capacity_kwh']
            if self.soc < min_soc:
                print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} battery low ({self.soc:.2f} kWh). Moving to charge.")
                yield self.env.process(self.move(config.FACILITY_LAYOUT['locations']['charging_station']))
                yield self.env.process(self.charge_full())
            
            print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} waiting for task at {self.location}. SoC: {self.soc:.2f} kWh")
            self.data_collector.start_downtime(self.vehicle_id, self.env.now)
            task = yield self.task_queue.get()
            self.data_collector.end_downtime(self.vehicle_id, self.env.now)
            
            yield self.env.process(self.execute_task(task))

    def run_fcfs(self):
        """Implements the First-Come, First-Served baseline (charge after every task)."""
        while True:
            print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} waiting for task at {self.location}. SoC: {self.soc:.2f} kWh")
            self.data_collector.start_downtime(self.vehicle_id, self.env.now)
            task = yield self.task_queue.get()
            self.data_collector.end_downtime(self.vehicle_id, self.env.now)

            yield self.env.process(self.execute_task(task))

            # After task, go charge
            print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} task complete. Moving to charge.")
            yield self.env.process(self.move(config.FACILITY_LAYOUT['locations']['charging_station']))
            yield self.env.process(self.charge_full())

    def run_intelligent(self):
        """Main process loop for the intelligent vehicle agent."""
        while True:
            if self.charging_schedule and self.env.now >= self.charging_schedule[0]:
                charge_time_to_execute = self.charging_schedule[0]
                yield self.env.process(self.move(config.FACILITY_LAYOUT['locations']['charging_station']))
                yield self.env.process(self.charge(60)) # Charge for 1 minute
                if self.charging_schedule and self.charging_schedule[0] == charge_time_to_execute:
                    self.charging_schedule.pop(0)
            else:
                print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} waiting for task or charging command. SoC: {self.soc:.2f} kWh")
                self.data_collector.start_downtime(self.vehicle_id, self.env.now)
                task = yield self.task_queue.get()
                self.data_collector.end_downtime(self.vehicle_id, self.env.now)
                yield self.env.process(self.execute_task(task))

    def execute_task(self, task):
        """Process for executing a single task."""
        print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} starting task {task['id']}: {task['origin']} -> {task['destination']}")
        # For now, we'll simulate movement to the charging station and then charging
        yield self.env.process(self.move(config.FACILITY_LAYOUT['locations']['charging_station']))
        
        # Simulate charging based on actual data
        # Ensure ChargingPower_kW is not NaN or inf before using
        charging_power_kw = task['charging_kwh'] / (task['charging_duration_minutes'] / 60) if task['charging_duration_minutes'] > 0 else config.VEHICLE_SPECS['charging_rate_kw']
        charging_duration_seconds = task['charging_duration_minutes'] * 60

        # Update vehicle's charging rate for this specific charge
        original_charging_rate = config.VEHICLE_SPECS['charging_rate_kw']
        config.VEHICLE_SPECS['charging_rate_kw'] = charging_power_kw

        yield self.env.process(self.charge(charging_duration_seconds))

        # Revert vehicle's charging rate
        config.VEHICLE_SPECS['charging_rate_kw'] = original_charging_rate

        print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} completed task {task['id']}.")
        self.data_collector.record_task_completion()

    def move(self, destination):
        """Process of moving from one location to another."""
        distance = np.linalg.norm(np.array(self.location) - np.array(destination)) / 100 # simple scaling for distance
        if distance == 0:
            return
            
        travel_time_hours = distance / config.VEHICLE_SPECS['speed_kmh']
        energy_consumed = travel_time_hours * config.VEHICLE_SPECS['energy_consumption_kwh_per_km']

        if self.soc < energy_consumed:
            print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} has insufficient energy to move to {destination}. THIS SHOULD NOT HAPPEN IN BASELINE.")
            self.soc = 0
            return

        print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} starting move to {destination}. Will take {travel_time_hours:.2f}h.")
        yield self.env.timeout(travel_time_hours * 3600)
        self.soc -= energy_consumed
        self.location = destination
        print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} arrived at {destination}. SoC is now {self.soc:.2f} kWh.")

    def charge(self, duration_seconds):
        """Process of charging the vehicle."""
        with self.charging_station.ports.request() as request:
            yield request
            print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} acquired charging port to charge for {duration_seconds}s.")
            
            energy_added = (duration_seconds / 3600) * config.VEHICLE_SPECS['charging_rate_kw']
            self.soc = min(self.soc + energy_added, config.VEHICLE_SPECS['battery_capacity_kwh'])
            self.data_collector.record_charge(energy_added)
            yield self.env.timeout(duration_seconds)
            print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} finished charging. SoC is now {self.soc:.2f} kWh.")

    def charge_full(self):
        """Process of charging the vehicle to full."""
        with self.charging_station.ports.request() as request:
            yield request
            print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} acquired charging port.")
            
            energy_needed = config.VEHICLE_SPECS['battery_capacity_kwh'] - self.soc
            time_to_charge_hours = energy_needed / config.VEHICLE_SPECS['charging_rate_kw']
            
            yield self.env.timeout(time_to_charge_hours * 3600)
            self.soc = config.VEHICLE_SPECS['battery_capacity_kwh']
            self.data_collector.record_charge(energy_needed)
            print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} finished charging. SoC is now {self.soc:.2f} kWh.")

class ChargingStation:
    """Models the charging station with a limited number of ports."""
    def __init__(self, env):
        self.env = env
        self.ports = simpy.Resource(env, capacity=config.NUM_CHARGING_PORTS)

def task_generator(env, task_queue, data_collector, charging_data_processor):
    """Generates tasks for the vehicles based on the charging data."""
    for index, row in charging_data_processor.charging_data.iterrows():
        yield env.timeout((row['Order creation time'].timestamp() - env.now) / 60) # Wait until the task creation time
        
        task_id = index # Use the DataFrame index as task_id
        # Use actual charging data for task details
        # For simplicity, we'll use a generic 'charging_task' type and pass relevant data
        task = {
            'id': task_id,
            'origin': 'charging_station', # Assuming tasks start at charging station for now
            'destination': 'charging_station', # Assuming tasks end at charging station for now
            'charging_kwh': row['Transaction power/kwh'],
            'charging_duration_minutes': row['ChargingDuration_minutes'],
            'start_time': row['Start Time'],
            'end_time': row['End Time'],
            'temperature': charging_data_processor.get_weather_data_for_time(row['Start Time'])['Temperature(℃)'] if charging_data_processor.get_weather_data_for_time(row['Start Time']) is not None else None
        }
        print(f"{env.now:.2f}: New task {task_id} generated.")
        data_collector.record_task_generation()
        yield task_queue.put(task)

def controller_process(env, vehicles, charging_data_processor):
    """Periodically solves the charging schedule and updates vehicles."""
    while True:
        print(f"{env.now:.2f}: Controller solving for optimal schedule...")
        schedule = optimization_controller.solve_charging_schedule(vehicles, env.now, charging_data_processor)
        for vehicle_id, charge_times in schedule.items():
            vehicles[vehicle_id].charging_schedule = [env.now + t * 60 for t in charge_times]
        
        yield env.timeout(config.TIME_STEP_MINUTES * 60)

def setup_simulation(strategy='uncontrolled'):
    """Sets up and returns the simulation environment."""
    env = simpy.Environment()
    data_collector = DataCollector()
    task_queue = simpy.Store(env)
    charging_station = ChargingStation(env)
    
    # Initialize ChargingDataProcessor
    charging_data_processor = ChargingDataProcessor(
        charging_data_path='/work3/xiuli/ev_charging/datasets/Charging_Data.csv',
        weather_data_path='/work3/xiuli/ev_charging/datasets/Weather_Data.csv'
    )

    vehicles = [Vehicle(env, i, task_queue, charging_station, data_collector, strategy) for i in range(config.NUM_VEHICLES)]
    env.process(task_generator(env, task_queue, data_collector, charging_data_processor))

    if strategy == 'intelligent':
        env.process(controller_process(env, vehicles, charging_data_processor))

    return env, data_collector
