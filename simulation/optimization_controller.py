import pulp
import pandas as pd
import numpy as np
from . import config

def get_electricity_price(time_of_day_hours):
    """Returns the electricity price for a given hour of the day."""
    for start, end, price in config.ELECTRICITY_PRICES_HOURLY:
        if start <= time_of_day_hours < end:
            return price
    return config.ELECTRICITY_PRICES_HOURLY[0][2] # Default to first price if no match (e.g., 23:59-00:00 transition)

def load_time_of_use_prices():
    """Load time-of-use electricity prices from CSV file."""
    try:
        price_df = pd.read_csv('datasets/Time-of-use_Price.csv')
        prices = []
        for _, row in price_df.iterrows():
            time_period = row['Time Period']
            price = row['Electricity Price(Yuan/kWh)']
            
            # Parse time period (e.g., "00:00-08:00")
            start_time, end_time = time_period.split('-')
            start_hour = int(start_time.split(':')[0])
            end_hour = int(end_time.split(':')[0])
            if end_hour == 0:  # Handle 24:00 as 0:00 next day
                end_hour = 24
            
            prices.append((start_hour, end_hour, price))
        
        return prices
    except Exception as e:
        print(f"Warning: Could not load price data: {e}")
        return config.ELECTRICITY_PRICES_HOURLY

# Update electricity prices from real data
config.ELECTRICITY_PRICES_HOURLY = load_time_of_use_prices()

def _assess_vehicle_states(vehicles, env_now):
    """Assess current state of all vehicles for predictive optimization."""
    states = {}
    for v in vehicles:
        # Calculate predicted energy consumption based on typical usage patterns
        predicted_consumption = _predict_vehicle_energy_consumption(v, env_now)

        # Assess urgency based on current SoC and predicted consumption
        min_soc = config.VEHICLE_SPECS['min_soc_percent'] / 100.0 * config.VEHICLE_SPECS['battery_capacity_kwh']
        urgency_score = max(0, (min_soc * 2 - v.soc) / config.VEHICLE_SPECS['battery_capacity_kwh'])

        states[v.vehicle_id] = {
            'current_soc': v.soc,
            'predicted_consumption': predicted_consumption,
            'urgency_score': urgency_score,
            'location': v.location,
            'available_for_charging': _is_vehicle_available_for_charging(v)
        }
    return states

def _predict_vehicle_energy_consumption(vehicle, env_now):
    """Predict energy consumption for a vehicle over the optimization horizon."""
    # Simple prediction based on historical average consumption
    # In a real implementation, this would use machine learning or statistical models
    avg_consumption_per_hour = 0.5  # kWh per hour (example)
    horizon_hours = config.OPTIMIZATION_HORIZON_MINUTES / 60
    return avg_consumption_per_hour * horizon_hours

def _is_vehicle_available_for_charging(vehicle):
    """Determine if a vehicle is available for charging."""
    # Check if vehicle is at or near charging station
    charging_location = config.FACILITY_LAYOUT['locations']['charging_station']
    distance_to_charging = np.linalg.norm(np.array(vehicle.location) - np.array(charging_location))
    return distance_to_charging < 50  # Within 50 meters of charging station

def _forecast_energy_demand(vehicles, env_now):
    """Forecast total energy demand over the optimization horizon."""
    total_demand = 0
    for v in vehicles:
        # Predict when vehicle will need charging based on current SoC and usage patterns
        min_soc = config.VEHICLE_SPECS['min_soc_percent'] / 100.0 * config.VEHICLE_SPECS['battery_capacity_kwh']
        if v.soc < min_soc * 1.5:  # If below 150% of minimum
            energy_needed = config.VEHICLE_SPECS['battery_capacity_kwh'] - v.soc
            total_demand += energy_needed
    return total_demand

def _forecast_electricity_prices(env_now):
    """Forecast electricity prices over the optimization horizon."""
    prices = []
    for t in range(config.OPTIMIZATION_HORIZON_MINUTES):
        future_time = env_now + t * 60
        hour_of_day = (future_time / 3600) % 24
        price = get_electricity_price(hour_of_day)
        prices.append(price)
    return prices

def _get_peak_penalty(timestamp):
    """Calculate peak demand penalty based on time of day and grid conditions."""
    hour_of_day = (timestamp / 3600) % 24

    # Higher penalties during peak hours (inspired by adaptive pricing)
    if 8 <= hour_of_day <= 11 or 17 <= hour_of_day <= 21:  # Peak hours
        return 0.5
    elif 12 <= hour_of_day <= 16:  # Mid-peak hours
        return 0.2
    else:  # Off-peak hours
        return 0.05

def _calculate_load_balancing_cost(charge_vars, vehicles):
    """Calculate cost for load balancing across time periods."""
    # Penalize large variations in total charging load
    # Note: PuLP doesn't support quadratic terms directly, so we use absolute differences
    load_variance_penalty = 0

    for t in range(config.OPTIMIZATION_HORIZON_MINUTES - 1):
        current_load = pulp.lpSum(charge_vars[v.vehicle_id, t] for v in vehicles)
        next_load = pulp.lpSum(charge_vars[v.vehicle_id, t + 1] for v in vehicles)
        # Use absolute difference approximation instead of squared difference
        # This is a linear approximation of the quadratic penalty
        load_diff = current_load - next_load
        load_variance_penalty += 0.01 * load_diff  # Simplified linear penalty

    return load_variance_penalty

def _calculate_adaptive_min_soc(vehicle, vehicle_state):
    """Calculate adaptive minimum SoC based on vehicle state and predicted usage."""
    base_min_soc = config.VEHICLE_SPECS['min_soc_percent'] / 100.0 * config.VEHICLE_SPECS['battery_capacity_kwh']

    # Increase minimum SoC based on urgency and predicted consumption
    urgency_factor = 1 + vehicle_state['urgency_score']
    consumption_buffer = vehicle_state['predicted_consumption'] * 0.2  # 20% buffer

    adaptive_min_soc = base_min_soc * urgency_factor + consumption_buffer

    # Cap at reasonable maximum (80% of battery capacity)
    max_min_soc = config.VEHICLE_SPECS['battery_capacity_kwh'] * 0.8

    return min(adaptive_min_soc, max_min_soc)

def solve_charging_schedule(vehicles, env_now, charging_data_processor):
    """
    Advanced Digital Twin-based optimization for EV charging scheduling.

    This implements a sophisticated optimization model inspired by the reference papers:
    - Adaptive pricing strategies from ref1.tex (Valogianni et al.)
    - Infrastructure optimization concepts from ref2.tex (Shi et al.)
    - Digital twin framework for real-time decision making
    - Rolling horizon optimization with predictive capabilities
    """

    if not vehicles:
        return {}

    try:
        # 1. Predictive State Assessment
        vehicle_states = _assess_vehicle_states(vehicles, env_now)
        demand_forecast = _forecast_energy_demand(vehicles, env_now)
        price_forecast = _forecast_electricity_prices(env_now)

        # 2. Create the optimization problem
        prob = pulp.LpProblem("Digital_Twin_EV_Charging_Optimization", pulp.LpMinimize)

        # 3. Enhanced Decision Variables
        # Continuous charging power variables
        charge_vars = pulp.LpVariable.dicts(
            "charge_power",
            ((v.vehicle_id, t) for v in vehicles for t in range(config.OPTIMIZATION_HORIZON_MINUTES)),
            lowBound=0,
            upBound=config.VEHICLE_SPECS['charging_rate_kw'] / 60,  # Max charging rate per minute
            cat='Continuous'
        )

        # Binary charging decision variables
        charging_decision = pulp.LpVariable.dicts(
            "is_charging",
            ((v.vehicle_id, t) for v in vehicles for t in range(config.OPTIMIZATION_HORIZON_MINUTES)),
            cat='Binary'
        )

        # Binary variables for charging port allocation
        port_allocation = pulp.LpVariable.dicts(
            "port_assigned",
            ((v.vehicle_id, t) for v in vehicles for t in range(config.OPTIMIZATION_HORIZON_MINUTES)),
            cat='Binary'
        )

        # 4. Advanced Multi-Objective Function
        # Inspired by adaptive pricing and infrastructure optimization from reference papers

        # 4.1 Electricity Cost (Time-of-Use Pricing)
        electricity_cost = pulp.lpSum(
            charge_vars[v.vehicle_id, t] * price_forecast[t]
            for v in vehicles for t in range(config.OPTIMIZATION_HORIZON_MINUTES)
        )

        # 4.2 Peak Demand Management (Grid Stability)
        peak_demand_penalty = pulp.lpSum(
            charging_decision[v.vehicle_id, t] * _get_peak_penalty(env_now + t * 60)
            for v in vehicles for t in range(config.OPTIMIZATION_HORIZON_MINUTES)
        )

        # 4.3 Vehicle Availability Optimization (Manufacturing Productivity)
        availability_cost = pulp.lpSum(
            vehicle_states[v.vehicle_id]['urgency_score'] *
            (1 - charging_decision[v.vehicle_id, t]) * 10  # Penalty for not charging urgent vehicles
            for v in vehicles for t in range(config.OPTIMIZATION_HORIZON_MINUTES)
        )

        # 4.4 Infrastructure Utilization Efficiency
        utilization_penalty = pulp.lpSum(
            (port_allocation[v.vehicle_id, t] - charging_decision[v.vehicle_id, t]) * 0.5
            for v in vehicles for t in range(config.OPTIMIZATION_HORIZON_MINUTES)
        )

        # 4.5 Load Balancing (Inspired by adaptive pricing concepts)
        load_balancing_cost = _calculate_load_balancing_cost(charge_vars, vehicles)

        # Combined objective function with weights
        prob += (
            1.0 * electricity_cost +           # Primary: minimize electricity cost
            0.3 * peak_demand_penalty +        # Secondary: avoid peak demand
            0.5 * availability_cost +          # Important: maintain vehicle availability
            0.2 * utilization_penalty +        # Efficiency: optimize infrastructure use
            0.1 * load_balancing_cost          # Stability: balance load across time
        )

        # 5. Advanced Constraint System

        # 5.1 Charging Power and Decision Linking
        for v in vehicles:
            for t in range(config.OPTIMIZATION_HORIZON_MINUTES):
                # Link continuous power with binary decision
                prob += charge_vars[v.vehicle_id, t] <= charging_decision[v.vehicle_id, t] * (config.VEHICLE_SPECS['charging_rate_kw'] / 60)
                # Link port allocation with charging decision
                prob += port_allocation[v.vehicle_id, t] >= charging_decision[v.vehicle_id, t]

        # 5.2 Infrastructure Capacity Constraints
        for t in range(config.OPTIMIZATION_HORIZON_MINUTES):
            # Physical charging port limitation
            prob += pulp.lpSum(port_allocation[v.vehicle_id, t] for v in vehicles) <= config.NUM_CHARGING_PORTS

            # Grid power limitation (prevent overloading)
            total_power = pulp.lpSum(charge_vars[v.vehicle_id, t] for v in vehicles)
            max_grid_power = config.NUM_CHARGING_PORTS * config.VEHICLE_SPECS['charging_rate_kw'] / 60 * 0.9  # 90% of max capacity
            prob += total_power <= max_grid_power

        # 5.3 Battery State Constraints
        for v in vehicles:
            current_soc = v.soc
            for t in range(config.OPTIMIZATION_HORIZON_MINUTES):
                # Update SoC considering charging and predicted consumption
                predicted_consumption = vehicle_states[v.vehicle_id]['predicted_consumption'] / config.OPTIMIZATION_HORIZON_MINUTES
                current_soc = current_soc + charge_vars[v.vehicle_id, t] - predicted_consumption

                # Battery capacity limits
                prob += current_soc <= config.VEHICLE_SPECS['battery_capacity_kwh']

                # Minimum SoC requirements (adaptive)
                min_soc_threshold = _calculate_adaptive_min_soc(v, vehicle_states[v.vehicle_id])
                prob += current_soc >= min_soc_threshold

        # 5.4 Vehicle Availability Constraints
        for v in vehicles:
            if not vehicle_states[v.vehicle_id]['available_for_charging']:
                # Vehicle not available for charging (e.g., on a task)
                for t in range(min(30, config.OPTIMIZATION_HORIZON_MINUTES)):  # Next 30 minutes
                    prob += charging_decision[v.vehicle_id, t] == 0

        # 5.5 Load Balancing Constraints (Grid Stability)
        avg_load_per_period = demand_forecast / config.OPTIMIZATION_HORIZON_MINUTES
        for t in range(config.OPTIMIZATION_HORIZON_MINUTES):
            total_load = pulp.lpSum(charge_vars[v.vehicle_id, t] for v in vehicles)
            # Limit deviation from average load (soft constraint through objective)
            prob += total_load <= avg_load_per_period * 1.5  # Max 150% of average

        # 6. Solve the Advanced Optimization Problem
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))  # 30-second time limit

        # 7. Extract and Process Results
        schedule = {v.vehicle_id: [] for v in vehicles}
        optimization_metrics = {}

        if pulp.LpStatus[prob.status] == 'Optimal':
            # Extract optimal charging schedule
            total_cost = 0
            total_energy = 0

            for v in vehicles:
                vehicle_schedule = []
                vehicle_energy = 0

                for t in range(config.OPTIMIZATION_HORIZON_MINUTES):
                    if pulp.value(charging_decision[v.vehicle_id, t]) == 1:
                        vehicle_schedule.append(t)
                        power = pulp.value(charge_vars[v.vehicle_id, t])
                        vehicle_energy += power
                        total_energy += power
                        total_cost += power * price_forecast[t]

                schedule[v.vehicle_id] = vehicle_schedule

            # Store optimization metrics for analysis
            optimization_metrics = {
                'status': 'optimal',
                'total_cost': total_cost,
                'total_energy': total_energy,
                'objective_value': pulp.value(prob.objective),
                'vehicles_scheduled': sum(1 for v_schedule in schedule.values() if v_schedule)
            }

            print(f"Optimal solution found: Cost=${total_cost:.2f}, Energy={total_energy:.2f}kWh")

        else:
            # Enhanced fallback with reason logging
            print(f"Optimization failed with status: {pulp.LpStatus[prob.status]}. Using intelligent fallback.")
            schedule = _intelligent_fallback_scheduling(vehicles, env_now, vehicle_states)
            optimization_metrics = {'status': 'fallback', 'reason': pulp.LpStatus[prob.status]}
    
    except Exception as e:
        print(f"Error in optimization: {e}. Using fallback scheduling.")
        schedule = _fallback_scheduling(vehicles, env_now)
    
    return schedule

def _intelligent_fallback_scheduling(vehicles, env_now, vehicle_states):
    """Intelligent fallback scheduling when optimization fails."""
    schedule = {v.vehicle_id: [] for v in vehicles}

    # Sort vehicles by urgency (most urgent first)
    vehicles_by_urgency = sorted(vehicles,
                                key=lambda v: vehicle_states[v.vehicle_id]['urgency_score'],
                                reverse=True)

    # Find off-peak periods for cost-effective charging
    off_peak_periods = _find_off_peak_periods(env_now)

    charging_slots_used = [0] * config.OPTIMIZATION_HORIZON_MINUTES

    for v in vehicles_by_urgency:
        vehicle_state = vehicle_states[v.vehicle_id]

        # Skip if vehicle is not available for charging
        if not vehicle_state['available_for_charging']:
            continue

        # Calculate charging need
        min_soc = config.VEHICLE_SPECS['min_soc_percent'] / 100.0 * config.VEHICLE_SPECS['battery_capacity_kwh']
        if v.soc < min_soc * 1.8:  # Schedule if below 180% of minimum

            # Determine charging duration needed
            energy_needed = min(config.VEHICLE_SPECS['battery_capacity_kwh'] - v.soc,
                              config.VEHICLE_SPECS['battery_capacity_kwh'] * 0.3)  # Max 30% charge
            charging_time_needed = int((energy_needed / config.VEHICLE_SPECS['charging_rate_kw']) * 60)  # minutes

            # Find best time slot (prefer off-peak, consider urgency)
            best_start_time = _find_best_charging_slot(
                charging_slots_used, charging_time_needed,
                off_peak_periods, vehicle_state['urgency_score']
            )

            if best_start_time is not None:
                # Schedule charging
                charging_duration = min(charging_time_needed,
                                      config.OPTIMIZATION_HORIZON_MINUTES - best_start_time)
                schedule[v.vehicle_id] = list(range(best_start_time, best_start_time + charging_duration))

                # Update slot usage
                for t in range(best_start_time, best_start_time + charging_duration):
                    charging_slots_used[t] += 1

    return schedule

def _find_off_peak_periods(env_now):
    """Identify off-peak periods for cost-effective charging."""
    off_peak_periods = []
    for t in range(config.OPTIMIZATION_HORIZON_MINUTES):
        hour_of_day = ((env_now + t * 60) / 3600) % 24
        if hour_of_day < 8 or hour_of_day > 22:  # Off-peak hours
            off_peak_periods.append(t)
    return off_peak_periods

def _find_best_charging_slot(charging_slots_used, duration_needed, off_peak_periods, urgency_score):
    """Find the best time slot for charging considering multiple factors."""
    best_slot = None
    best_score = float('-inf')

    for start_time in range(config.OPTIMIZATION_HORIZON_MINUTES - duration_needed + 1):
        # Check if slot is available
        slot_available = all(charging_slots_used[t] < config.NUM_CHARGING_PORTS
                           for t in range(start_time, start_time + duration_needed))

        if slot_available:
            # Calculate slot score
            score = 0

            # Prefer off-peak periods
            off_peak_overlap = sum(1 for t in range(start_time, start_time + duration_needed)
                                 if t in off_peak_periods)
            score += off_peak_overlap * 2

            # For urgent vehicles, prefer earlier slots
            if urgency_score > 0.5:
                score += (config.OPTIMIZATION_HORIZON_MINUTES - start_time) * urgency_score

            # Prefer less congested slots
            congestion = sum(charging_slots_used[t] for t in range(start_time, start_time + duration_needed))
            score -= congestion * 0.5

            if score > best_score:
                best_score = score
                best_slot = start_time

    return best_slot

def _fallback_scheduling(vehicles, env_now):
    """Simple fallback for backward compatibility."""
    return _intelligent_fallback_scheduling(vehicles, env_now, _assess_vehicle_states(vehicles, env_now))
