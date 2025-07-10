import pulp
from . import config

def get_electricity_price(time_of_day_hours):
    """Returns the electricity price for a given hour of the day."""
    for start, end, price in config.ELECTRICITY_PRICES_HOURLY:
        if start <= time_of_day_hours < end:
            return price
    return config.ELECTRICITY_PRICES_HOURLY[0][2] # Default to first price if no match (e.g., 23:59-00:00 transition)

def solve_charging_schedule(vehicles, env_now, charging_data_processor):
    """Solves the optimization problem to determine the optimal charging schedule."""
    
    # 1. Create the problem
    prob = pulp.LpProblem("EV_Charging_Scheduling", pulp.LpMinimize)

    # 2. Define Decision Variables
    # charge_vars[i][t] is a binary variable indicating if vehicle i charges at time t
    charge_vars = pulp.LpVariable.dicts(
        "charge", 
        ((v.vehicle_id, t) for v in vehicles for t in range(config.OPTIMIZATION_HORIZON_MINUTES)),
        cat='Binary'
    )

    # 3. Define Objective Function
    # Minimize the total electricity cost of charging and vehicle downtime.
    cost = pulp.lpSum(
        charge_vars[v.vehicle_id, t] *
        (config.VEHICLE_SPECS['charging_rate_kw'] / 60) *
        get_electricity_price((env_now + t * 60) / 3600 % 24) +
        charge_vars[v.vehicle_id, t] *
        (config.VEHICLE_DOWNTIME_COST_PER_HOUR / 60)
        for v in vehicles for t in range(config.OPTIMIZATION_HORIZON_MINUTES)
    )
    prob += cost

    # 4. Define Constraints
    # Constraint: Each charging port can only charge one vehicle at a time.
    for t in range(config.OPTIMIZATION_HORIZON_MINUTES):
        prob += pulp.lpSum(charge_vars[v.vehicle_id, t] for v in vehicles) <= config.NUM_CHARGING_PORTS

    # Constraint: Ensure vehicles have enough charge to complete their next task (simplified)
    for v in vehicles:
        # This is a simplified placeholder. A full implementation would need to predict future tasks.
        # For now, we assume a vehicle needs at least 50% SoC.
        # We will use the actual charging data to determine the required SoC.
        random_charging_event = charging_data_processor.get_random_charging_event()
        required_soc = random_charging_event['Transaction power/kwh']
        energy_per_minute = config.VEHICLE_SPECS['charging_rate_kw'] / 60
        
        prob += v.soc + pulp.lpSum(charge_vars[v.vehicle_id, t] * energy_per_minute 
                                for t in range(config.OPTIMIZATION_HORIZON_MINUTES)) >= required_soc

    # 5. Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=0)) # msg=0 suppresses solver output

    # 6. Extract the results
    schedule = {v.vehicle_id: [] for v in vehicles}
    if pulp.LpStatus[prob.status] == 'Optimal':
        for v in vehicles:
            for t in range(config.OPTIMIZATION_HORIZON_MINUTES):
                if pulp.value(charge_vars[v.vehicle_id, t]) == 1:
                    schedule[v.vehicle_id].append(t)
    
    return schedule
