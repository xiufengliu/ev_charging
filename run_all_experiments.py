#!/usr/bin/env python3
"""
Experimental Results Runner for Digital Twin EV Charging System
===============================================================

This script runs actual simulations to generate experimental results that validate 
the performance claims of the Digital Twin EV charging optimization system.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add simulation module to path
sys.path.append('.')

def run_sensor_noise_experiments():
    """Run actual sensor noise experimental simulations."""
    print("Running sensor noise experimental simulations...")
    
    try:
        from simulation.experiments.sensor_noise_experiments import SensorNoiseExperiments
        
        # Initialize and run experiments
        experiments = SensorNoiseExperiments()
        experiments.run_comprehensive_noise_analysis(num_replications=30)
        
        print("Sensor noise experiments completed successfully.")
        return True
        
    except ImportError as e:
        print(f"Error importing sensor noise experiments: {e}")
        return False
    except Exception as e:
        print(f"Error running sensor noise experiments: {e}")
        return False

def run_lstm_prediction_experiments():
    """Run actual LSTM prediction experimental simulations."""
    print("Running LSTM prediction experimental simulations...")
    
    try:
        from simulation.experiments.lstm_prediction_experiments import LSTMPredictionExperiments
        
        # Initialize and run experiments
        experiments = LSTMPredictionExperiments()
        experiments.run_comprehensive_prediction_analysis(num_replications=30)
        
        print("LSTM prediction experiments completed successfully.")
        return True
        
    except ImportError as e:
        print(f"Error importing LSTM prediction experiments: {e}")
        return False
    except Exception as e:
        print(f"Error running LSTM prediction experiments: {e}")
        return False

def run_communication_experiments():
    """Run actual communication robustness experimental simulations."""
    print("Running communication robustness experimental simulations...")
    
    try:
        from simulation.experiments.communication_experiments import CommunicationExperiments
        
        # Initialize and run experiments
        experiments = CommunicationExperiments()
        experiments.run_comprehensive_communication_analysis(num_replications=30)
        
        print("Communication experiments completed successfully.")
        return True
        
    except ImportError as e:
        print(f"Error importing communication experiments: {e}")
        return False
    except Exception as e:
        print(f"Error running communication experiments: {e}")
        return False

def run_carbon_emission_experiments():
    """Run actual carbon emission optimization experimental simulations."""
    print("Running carbon emission optimization experimental simulations...")
    
    try:
        from simulation.experiments.carbon_emission_experiments import CarbonEmissionExperiments
        
        # Initialize and run experiments
        experiments = CarbonEmissionExperiments()
        experiments.run_comprehensive_carbon_analysis(num_replications=30)
        
        print("Carbon emission experiments completed successfully.")
        return True
        
    except ImportError as e:
        print(f"Error importing carbon emission experiments: {e}")
        return False
    except Exception as e:
        print(f"Error running carbon emission experiments: {e}")
        return False

def run_sensitivity_analysis_experiments():
    """Run actual sensitivity analysis experimental simulations."""
    print("Running sensitivity analysis experimental simulations...")
    
    try:
        from simulation.experiments.sensitivity_analysis_experiments import SensitivityAnalysisExperiments
        
        # Initialize and run experiments
        experiments = SensitivityAnalysisExperiments()
        experiments.run_comprehensive_sensitivity_analysis(num_replications=30)
        
        print("Sensitivity analysis experiments completed successfully.")
        return True
        
    except ImportError as e:
        print(f"Error importing sensitivity analysis experiments: {e}")
        return False
    except Exception as e:
        print(f"Error running sensitivity analysis experiments: {e}")
        return False

def run_comprehensive_experiments():
    """Run comprehensive experimental framework."""
    print("Running comprehensive experimental framework...")
    
    try:
        from simulation.experimental_framework import run_journal_quality_experiments
        
        # Run the comprehensive experiments
        framework = run_journal_quality_experiments()
        
        print("Comprehensive experiments completed successfully.")
        return True
        
    except ImportError as e:
        print(f"Error importing experimental framework: {e}")
        return False
    except Exception as e:
        print(f"Error running comprehensive experiments: {e}")
        return False

def main():
    """Main function to run all experimental simulations."""
    print("="*80)
    print("DIGITAL TWIN EV CHARGING SYSTEMS - EXPERIMENTAL RESULTS RUNNER")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # List of experiments to run
    experiments = [
        ("Sensor Noise Analysis", run_sensor_noise_experiments),
        ("LSTM Prediction Analysis", run_lstm_prediction_experiments),
        ("Communication Robustness", run_communication_experiments),
        ("Carbon Emission Optimization", run_carbon_emission_experiments),
        ("Sensitivity Analysis", run_sensitivity_analysis_experiments),
        ("Comprehensive Framework", run_comprehensive_experiments)
    ]
    
    results = {}
    
    for exp_name, exp_function in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {exp_name}")
        print(f"{'='*60}")
        
        try:
            success = exp_function()
            results[exp_name] = "SUCCESS" if success else "FAILED"
        except Exception as e:
            print(f"Error in {exp_name}: {e}")
            results[exp_name] = "ERROR"
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENTAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    for exp_name, status in results.items():
        status_symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{status_symbol} {exp_name}: {status}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
