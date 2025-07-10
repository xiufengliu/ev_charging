#!/usr/bin/env python3
"""
Main script to run comprehensive experiments for the Digital Twin EV Charging Systems research.

This script executes the full experimental framework designed for submission to the
Journal of Manufacturing Systems, including:
- Multiple charging strategies comparison
- Statistical analysis
- Performance visualization
- LaTeX table generation for the paper
"""

import sys
import os
import traceback
from datetime import datetime

# Add simulation module to path
sys.path.append('.')

def main():
    """Main function to run all experiments."""
    
    print("="*80)
    print("DIGITAL TWIN EV CHARGING SYSTEMS - RESEARCH EXPERIMENTS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Import after adding to path
        from simulation.experimental_framework import run_journal_quality_experiments
        
        print("Starting comprehensive experimental framework...")
        print("This may take several minutes to complete...")
        print()
        
        # Run the comprehensive experiments
        framework = run_journal_quality_experiments()
        
        print()
        print("="*80)
        print("EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Results saved to: results_journal_submission/")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Print summary of generated files
        results_dir = "results_journal_submission"
        if os.path.exists(results_dir):
            print(f"\nGenerated files:")
            for file in os.listdir(results_dir):
                print(f"  - {file}")
        
        return True
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Attempting to run basic simulation test...")
        return run_basic_test()
        
    except Exception as e:
        print(f"Error during experiments: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        print("Attempting to run basic simulation test...")
        return run_basic_test()

def run_basic_test():
    """Run a basic test of the simulation framework."""
    
    print("\n" + "="*60)
    print("RUNNING BASIC SIMULATION TEST")
    print("="*60)
    
    try:
        from simulation import main as sim_main
        
        print("Testing uncontrolled charging strategy...")
        sim_main.run_simulation(strategy='uncontrolled')
        
        print("\nBasic test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Basic test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def check_dependencies():
    """Check if all required dependencies are available."""
    
    print("Checking dependencies...")
    
    required_packages = [
        'simpy', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'scipy', 'pulp'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    print("All dependencies available!")
    return True

def check_data_files():
    """Check if required data files are available."""
    
    print("\nChecking data files...")
    
    required_files = [
        'datasets/Charging_Data.csv',
        'datasets/Weather_Data.csv', 
        'datasets/Time-of-use_Price.csv'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing data files: {missing_files}")
        return False
    
    print("All data files available!")
    return True

if __name__ == "__main__":
    print("Digital Twin EV Charging Systems - Experimental Framework")
    print("========================================================")
    
    # Check prerequisites
    if not check_dependencies():
        print("Please install missing dependencies before running experiments.")
        sys.exit(1)
    
    if not check_data_files():
        print("Please ensure all required data files are available.")
        sys.exit(1)
    
    # Run experiments
    success = main()
    
    if success:
        print("\n🎉 All experiments completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Experiments failed. Please check the error messages above.")
        sys.exit(1)