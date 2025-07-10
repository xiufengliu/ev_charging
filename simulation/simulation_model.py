import simpy
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
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

        self.weather_data['Date'] = pd.to_datetime(self.weather_data['Date'], format='%Y%m%d')

        # Handle missing values more carefully using modern pandas methods
        median_power = self.charging_data['Transaction power/kwh'].median()
        self.charging_data = self.charging_data.copy()
        self.charging_data['Transaction power/kwh'] = self.charging_data['Transaction power/kwh'].fillna(median_power)
        self.charging_data = self.charging_data.fillna(0)
        self.weather_data = self.weather_data.ffill()

        # Calculate charging duration and power with error handling
        self.charging_data['ChargingDuration_minutes'] = (self.charging_data['End Time'] - self.charging_data['Start Time']).dt.total_seconds() / 60

        # Avoid division by zero and handle invalid durations
        valid_duration_mask = self.charging_data['ChargingDuration_minutes'] > 0
        self.charging_data.loc[valid_duration_mask, 'ChargingPower_kW'] = (
            self.charging_data.loc[valid_duration_mask, 'Transaction power/kwh'] /
            (self.charging_data.loc[valid_duration_mask, 'ChargingDuration_minutes'] / 60)
        )

        # Fill invalid power values with default charging rate
        self.charging_data['ChargingPower_kW'] = self.charging_data['ChargingPower_kW'].fillna(config.VEHICLE_SPECS['charging_rate_kw'])

        # Remove outliers and invalid data
        self.charging_data = self.charging_data[
            (self.charging_data['ChargingDuration_minutes'] > 0) &
            (self.charging_data['ChargingDuration_minutes'] < 1440) &  # Less than 24 hours
            (self.charging_data['Transaction power/kwh'] > 0) &
            (self.charging_data['Transaction power/kwh'] < 100)  # Reasonable energy amounts
        ]

        # Sort by order creation time for proper task generation
        self.charging_data = self.charging_data.sort_values('Order creation time').reset_index(drop=True)

        print(f"Processed {len(self.charging_data)} valid charging records")

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
        self.total_energy_charged = 0
        self.tasks_completed = 0
        self.tasks_generated = 0
        self.vehicle_downtime = {i: 0 for i in range(config.NUM_VEHICLES)}
        self.last_downtime_start = {i: 0 for i in range(config.NUM_VEHICLES)}

    def record_charge(self, energy):
        self.total_energy_charged += energy

    def record_energy_consumption(self, energy):
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
        print(f"Total Energy Consumed (Movement): {self.total_energy_consumed:.2f} kWh")
        print(f"Total Energy Charged: {self.total_energy_charged:.2f} kWh")
        print(f"Net Energy Balance: {self.total_energy_charged - self.total_energy_consumed:.2f} kWh")
        print(f"Total Tasks Generated: {self.tasks_generated}")
        print(f"Total Tasks Completed: {self.tasks_completed}")
        completion_rate = (self.tasks_completed / max(self.tasks_generated, 1)) * 100
        print(f"Task Completion Rate: {completion_rate:.1f}%")
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
        """Process for executing a single task in the manufacturing environment."""
        print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} starting task {task['id']}: {task['origin']} -> {task['destination']}")

        # Move to task origin if not already there
        origin_location = config.FACILITY_LAYOUT['locations'][task['origin']]
        if self.location != origin_location:
            yield self.env.process(self.move(origin_location))

        # Simulate task execution time (loading/unloading, processing)
        task_execution_time = random.uniform(300, 900)  # 5-15 minutes
        print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} executing task at {task['origin']} for {task_execution_time/60:.1f} minutes")
        yield self.env.timeout(task_execution_time)

        # Move to destination
        destination_location = config.FACILITY_LAYOUT['locations'][task['destination']]
        yield self.env.process(self.move(destination_location))

        # Simulate unloading/delivery time
        unload_time = random.uniform(120, 300)  # 2-5 minutes
        print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} unloading at {task['destination']} for {unload_time/60:.1f} minutes")
        yield self.env.timeout(unload_time)

        print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} completed task {task['id']}.")
        self.data_collector.record_task_completion()

    def move(self, destination):
        """Process of moving from one location to another."""
        if isinstance(destination, str):
            destination = config.FACILITY_LAYOUT['locations'][destination]

        distance_m = np.linalg.norm(np.array(self.location) - np.array(destination))
        if distance_m < 1:  # Already at destination
            return

        distance_km = distance_m / 1000  # Convert to kilometers
        travel_time_hours = distance_km / config.VEHICLE_SPECS['speed_kmh']
        energy_consumed = distance_km * config.VEHICLE_SPECS['energy_consumption_kwh_per_km']

        if self.soc < energy_consumed:
            print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} has insufficient energy to move to {destination}. Emergency charging needed!")
            # Emergency charging to minimum level
            yield self.env.process(self.move(config.FACILITY_LAYOUT['locations']['charging_station']))
            yield self.env.process(self.charge_to_minimum())
            # Retry the move
            yield self.env.process(self.move(destination))
            return

        print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} moving to {destination} ({distance_km:.2f}km, {travel_time_hours*60:.1f}min)")
        yield self.env.timeout(travel_time_hours * 3600)
        self.soc -= energy_consumed
        self.data_collector.record_energy_consumption(energy_consumed)  # Track energy consumption
        self.location = destination
        print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} arrived. SoC: {self.soc:.2f} kWh")

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
            print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} acquired charging port for full charge.")

            energy_needed = config.VEHICLE_SPECS['battery_capacity_kwh'] - self.soc
            if energy_needed <= 0:
                print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} already fully charged.")
                return

            time_to_charge_hours = energy_needed / config.VEHICLE_SPECS['charging_rate_kw']

            yield self.env.timeout(time_to_charge_hours * 3600)
            self.soc = config.VEHICLE_SPECS['battery_capacity_kwh']
            self.data_collector.record_charge(energy_needed)
            print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} finished full charge. SoC: {self.soc:.2f} kWh.")

    def charge_to_minimum(self):
        """Process of charging the vehicle to minimum safe level."""
        with self.charging_station.ports.request() as request:
            yield request
            min_soc = config.VEHICLE_SPECS['min_soc_percent'] / 100.0 * config.VEHICLE_SPECS['battery_capacity_kwh'] * 1.5
            energy_needed = max(0, min_soc - self.soc)

            if energy_needed <= 0:
                print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} already above minimum charge level.")
                return

            time_to_charge_hours = energy_needed / config.VEHICLE_SPECS['charging_rate_kw']

            print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} emergency charging for {time_to_charge_hours*60:.1f} minutes.")
            yield self.env.timeout(time_to_charge_hours * 3600)
            self.soc = min(self.soc + energy_needed, config.VEHICLE_SPECS['battery_capacity_kwh'])
            self.data_collector.record_charge(energy_needed)
            print(f"{self.env.now:.2f}: Vehicle {self.vehicle_id} emergency charge complete. SoC: {self.soc:.2f} kWh.")

class ChargingStation:
    """Models the charging station with a limited number of ports."""
    def __init__(self, env):
        self.env = env
        self.ports = simpy.Resource(env, capacity=config.NUM_CHARGING_PORTS)

def task_generator(env, task_queue, data_collector, charging_data_processor):
    """Generates tasks for the vehicles based on manufacturing workload patterns."""
    # Use a more realistic task generation approach for manufacturing environment
    # Generate tasks at regular intervals based on the configured arrival rate

    task_id = 0
    locations = list(config.FACILITY_LAYOUT['locations'].keys())

    while True:
        # Wait for next task arrival (Poisson process)
        inter_arrival_time = np.random.exponential(3600 / config.TASK_ARRIVAL_RATE_PER_HOUR)
        yield env.timeout(inter_arrival_time)

        # Sample a charging event from the data to get realistic charging parameters
        charging_event = charging_data_processor.get_random_charging_event()

        # Generate a manufacturing task with realistic parameters
        origin = random.choice(locations)
        destination = random.choice([loc for loc in locations if loc != origin])

        task = {
            'id': task_id,
            'origin': origin,
            'destination': destination,
            'charging_kwh': charging_event['Transaction power/kwh'],
            'charging_duration_minutes': charging_event['ChargingDuration_minutes'],
            'start_time': env.now,
            'priority': random.choice(['low', 'medium', 'high']),
            'temperature': charging_event.get('Temperature(℃)', 20.0)  # Default temperature
        }

        print(f"{env.now:.2f}: New task {task_id} generated: {origin} -> {destination}")
        data_collector.record_task_generation()
        task_queue.put(task)
        task_id += 1

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
        charging_data_path='datasets/Charging_Data.csv',
        weather_data_path='datasets/Weather_Data.csv'
    )

    vehicles = [Vehicle(env, i, task_queue, charging_station, data_collector, strategy) for i in range(config.NUM_VEHICLES)]
    env.process(task_generator(env, task_queue, data_collector, charging_data_processor))

    if strategy == 'intelligent':
        env.process(controller_process(env, vehicles, charging_data_processor))

    return env, data_collector
