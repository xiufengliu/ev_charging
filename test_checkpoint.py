#!/usr/bin/env python3
"""
Test script to verify checkpoint functionality works correctly.
"""

import sys
import os
from datetime import datetime

# Add simulation module to path
sys.path.append('.')

def test_checkpoint_functionality():
    """Test that checkpoint save/load works correctly."""
    
    print("="*60)
    print("TESTING CHECKPOINT FUNCTIONALITY")
    print("="*60)
    
    try:
        from simulation.experimental_framework import ExperimentalFramework
        
        # Create framework with checkpoints enabled
        framework = ExperimentalFramework(output_dir="test_checkpoint_results", enable_checkpoints=True)
        
        # Test checkpoint save
        print("Testing checkpoint save...")
        test_scenarios = {'test_scenario': {'num_vehicles': 2, 'simulation_hours': 1}}
        framework.save_checkpoint('test_scenario', 'uncontrolled', 5, test_scenarios)
        
        # Test checkpoint load
        print("Testing checkpoint load...")
        checkpoint = framework.load_checkpoint()
        
        if checkpoint:
            print(f"✓ Checkpoint loaded successfully!")
            print(f"  - Scenario: {checkpoint.get('current_scenario')}")
            print(f"  - Strategy: {checkpoint.get('current_strategy')}")
            print(f"  - Replication: {checkpoint.get('current_replication')}")
            print(f"  - Timestamp: {checkpoint.get('timestamp')}")
        else:
            print("✗ Failed to load checkpoint")
            return False
        
        # Clean up
        framework.clear_checkpoint()
        if os.path.exists("test_checkpoint_results"):
            import shutil
            shutil.rmtree("test_checkpoint_results")
        
        print("✓ Checkpoint functionality test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Checkpoint test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def check_existing_checkpoint():
    """Check if there's an existing checkpoint from the running experiment."""
    
    print("\n" + "="*60)
    print("CHECKING FOR EXISTING CHECKPOINT")
    print("="*60)
    
    checkpoint_file = "results_journal_submission/experiment_checkpoint.pkl"
    
    if os.path.exists(checkpoint_file):
        print(f"✓ Checkpoint file found: {checkpoint_file}")
        
        try:
            from simulation.experimental_framework import ExperimentalFramework
            framework = ExperimentalFramework(output_dir="results_journal_submission", enable_checkpoints=True)
            checkpoint = framework.load_checkpoint()
            
            if checkpoint:
                print("Checkpoint details:")
                print(f"  - Current scenario: {checkpoint.get('current_scenario')}")
                print(f"  - Current strategy: {checkpoint.get('current_strategy')}")
                print(f"  - Current replication: {checkpoint.get('current_replication')}")
                print(f"  - Saved at: {checkpoint.get('timestamp')}")
                
                # Calculate progress
                scenarios = checkpoint.get('scenarios', {})
                total_scenarios = len(scenarios)
                total_strategies = 3  # uncontrolled, fcfs, intelligent
                total_replications = 50
                total_runs = total_scenarios * total_strategies * total_replications
                
                scenario_names = list(scenarios.keys())
                current_scenario_idx = scenario_names.index(checkpoint['current_scenario']) if checkpoint['current_scenario'] in scenario_names else 0
                strategy_names = ['uncontrolled', 'fcfs', 'intelligent']
                current_strategy_idx = strategy_names.index(checkpoint['current_strategy']) if checkpoint['current_strategy'] in strategy_names else 0
                current_replication = checkpoint.get('current_replication', 0)
                
                completed_runs = (current_scenario_idx * total_strategies * total_replications + 
                                current_strategy_idx * total_replications + 
                                current_replication)
                
                progress_percent = (completed_runs / total_runs) * 100
                
                print(f"  - Progress: {completed_runs}/{total_runs} runs ({progress_percent:.1f}%)")
                print(f"  - Estimated remaining time: {(total_runs - completed_runs) * 3 / 60:.1f} hours")
                
                return True
            else:
                print("✗ Could not load checkpoint data")
                return False
                
        except Exception as e:
            print(f"✗ Error reading checkpoint: {e}")
            return False
    else:
        print("No existing checkpoint found.")
        print("The experiment will start from the beginning if run.")
        return False

if __name__ == "__main__":
    print("Digital Twin EV Charging Systems - Checkpoint Test")
    print("=================================================")
    
    # Test checkpoint functionality
    if test_checkpoint_functionality():
        print("\n✅ Checkpoint system is working correctly!")
    else:
        print("\n❌ Checkpoint system has issues!")
        sys.exit(1)
    
    # Check for existing checkpoint from running experiment
    check_existing_checkpoint()
    
    print("\n" + "="*60)
    print("CHECKPOINT SYSTEM READY")
    print("="*60)
    print("The experimental framework now supports:")
    print("  ✓ Automatic checkpoint saving every 5 replications")
    print("  ✓ Resume from last checkpoint if interrupted")
    print("  ✓ Progress tracking and estimation")
    print("  ✓ Automatic cleanup after successful completion")
    print("\nYour running experiment will now be protected against crashes!")
