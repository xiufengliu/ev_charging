#!/usr/bin/env python3
"""
Monitor simulation progress and provide status updates.
"""

import os
import pickle
import time
from datetime import datetime, timedelta

def load_checkpoint():
    """Load the current checkpoint if it exists."""
    checkpoint_file = "results_journal_submission/experiment_checkpoint.pkl"
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    return None

def calculate_progress(checkpoint):
    """Calculate overall progress based on checkpoint data."""
    if not checkpoint:
        return 0, 0, 0
    
    scenarios = checkpoint.get('scenarios', {})
    total_scenarios = len(scenarios)
    total_strategies = 3  # uncontrolled, fcfs, intelligent
    total_replications = 50
    total_runs = total_scenarios * total_strategies * total_replications
    
    scenario_names = list(scenarios.keys())
    current_scenario = checkpoint.get('current_scenario', '')
    current_strategy = checkpoint.get('current_strategy', '')
    current_replication = checkpoint.get('current_replication', 0)
    
    if current_scenario in scenario_names:
        current_scenario_idx = scenario_names.index(current_scenario)
    else:
        current_scenario_idx = 0
    
    strategy_names = ['uncontrolled', 'fcfs', 'intelligent']
    if current_strategy in strategy_names:
        current_strategy_idx = strategy_names.index(current_strategy)
    else:
        current_strategy_idx = 0
    
    completed_runs = (current_scenario_idx * total_strategies * total_replications + 
                     current_strategy_idx * total_replications + 
                     current_replication)
    
    progress_percent = (completed_runs / total_runs) * 100
    
    return completed_runs, total_runs, progress_percent

def estimate_remaining_time(checkpoint, progress_percent):
    """Estimate remaining time based on current progress."""
    if not checkpoint or progress_percent <= 0:
        return "Unknown"
    
    timestamp_str = checkpoint.get('timestamp', '')
    if not timestamp_str:
        return "Unknown"
    
    try:
        start_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        current_time = datetime.now()
        elapsed_time = current_time - start_time
        
        if progress_percent >= 100:
            return "Completed"
        
        total_estimated_time = elapsed_time * (100 / progress_percent)
        remaining_time = total_estimated_time - elapsed_time
        
        hours = int(remaining_time.total_seconds() // 3600)
        minutes = int((remaining_time.total_seconds() % 3600) // 60)
        
        return f"{hours}h {minutes}m"
    except Exception as e:
        return f"Error: {e}"

def check_screen_session():
    """Check if the screen session is still running."""
    try:
        result = os.system("screen -list | grep -q ev_simulation")
        return result == 0
    except:
        return False

def main():
    """Main monitoring function."""
    print("="*60)
    print("EV CHARGING SIMULATION MONITOR")
    print("="*60)
    
    # Check if screen session is running
    if check_screen_session():
        print("✓ Screen session 'ev_simulation' is running")
    else:
        print("✗ Screen session 'ev_simulation' not found")
        print("The simulation may have completed or crashed.")
    
    # Load checkpoint and calculate progress
    checkpoint = load_checkpoint()
    
    if checkpoint:
        completed_runs, total_runs, progress_percent = calculate_progress(checkpoint)
        remaining_time = estimate_remaining_time(checkpoint, progress_percent)
        
        print(f"\nCurrent Status:")
        print(f"  - Scenario: {checkpoint.get('current_scenario', 'Unknown')}")
        print(f"  - Strategy: {checkpoint.get('current_strategy', 'Unknown')}")
        print(f"  - Replication: {checkpoint.get('current_replication', 0)}/50")
        print(f"  - Overall Progress: {completed_runs}/{total_runs} ({progress_percent:.1f}%)")
        print(f"  - Estimated Remaining Time: {remaining_time}")
        print(f"  - Last Updated: {checkpoint.get('timestamp', 'Unknown')}")
        
        # Progress bar
        bar_length = 40
        filled_length = int(bar_length * progress_percent / 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"  - Progress: |{bar}| {progress_percent:.1f}%")
        
        # Check if results are available
        results = checkpoint.get('results', {})
        if results:
            print(f"\nCompleted Scenarios: {len(results)}")
            for scenario_name, scenario_results in results.items():
                print(f"  - {scenario_name}: {len(scenario_results)} strategies completed")
    else:
        print("No checkpoint found. Simulation may not have started yet.")
    
    # Check for completed results
    results_dir = "results_journal_submission"
    if os.path.exists(results_dir):
        files = [f for f in os.listdir(results_dir) if f.endswith(('.csv', '.json', '.pdf', '.tex'))]
        if files:
            print(f"\nCompleted Result Files:")
            for file in sorted(files):
                file_path = os.path.join(results_dir, file)
                size = os.path.getsize(file_path)
                print(f"  - {file} ({size} bytes)")
        else:
            print(f"\nNo completed result files found yet.")
    
    print("\n" + "="*60)
    print("To view live simulation output:")
    print("  screen -r ev_simulation")
    print("To detach from screen session: Ctrl+A, then D")
    print("="*60)

if __name__ == "__main__":
    main()
