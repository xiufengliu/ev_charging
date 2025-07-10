from . import config
from . import simulation_model

def run_simulation(strategy):
    """Runs the main simulation experiment for a given strategy."""
    print(f"--- Starting Simulation: {strategy.upper()} Strategy ---")
    
    # Setup the simulation environment
    env, data_collector = simulation_model.setup_simulation(strategy=strategy)
    
    # Run the simulation
    simulation_duration_seconds = config.SIMULATION_TIME_HOURS * 3600
    env.run(until=simulation_duration_seconds)
    
    print(f"--- Simulation Finished: {strategy.upper()} Strategy ---")
    data_collector.report()

if __name__ == '__main__':
    run_simulation(strategy='uncontrolled')
    run_simulation(strategy='fcfs')
    run_simulation(strategy='intelligent')
