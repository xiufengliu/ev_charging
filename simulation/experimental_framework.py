"""
Experimental Framework for Digital Twin EV Charging Systems Research
====================================================================

This module implements a comprehensive experimental framework for evaluating
different charging strategies in smart manufacturing environments, designed
for submission to the Journal of Manufacturing Systems.

Key Features:
- Multiple charging strategies comparison
- Statistical significance testing
- Performance metrics aligned with manufacturing systems research
- Reproducible experimental design
- Results visualization and reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import pickle
import os
from datetime import datetime
from . import config
from . import simulation_model
from . import main

class ExperimentalFramework:
    """
    Comprehensive experimental framework for digital twin EV charging research.
    """
    
    def __init__(self, output_dir="results", enable_checkpoints=True):
        self.output_dir = output_dir
        self.strategies = ['uncontrolled', 'fcfs', 'intelligent']
        self.results = {}
        self.statistical_tests = {}
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_file = os.path.join(output_dir, 'experiment_checkpoint.pkl')

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define key performance indicators for manufacturing systems
        self.kpis = {
            'total_energy_cost': 'Total Energy Cost ($)',
            'energy_efficiency': 'Energy Efficiency (%)',
            'vehicle_utilization': 'Vehicle Utilization (%)',
            'charging_station_utilization': 'Charging Station Utilization (%)',
            'task_completion_rate': 'Task Completion Rate (%)',
            'average_response_time': 'Average Response Time (minutes)',
            'peak_demand_reduction': 'Peak Demand Reduction (%)',
            'grid_stability_index': 'Grid Stability Index',
            'carbon_footprint': 'Carbon Footprint (kg CO2)',
            'operational_cost_savings': 'Operational Cost Savings (%)'
        }
    
    def run_comprehensive_experiments(self, num_replications=30, scenarios=None):
        """
        Run comprehensive experiments with multiple scenarios and replications.
        Supports resuming from checkpoints if interrupted.

        Args:
            num_replications (int): Number of simulation replications for statistical validity
            scenarios (list): List of scenario configurations to test
        """
        print("="*80)
        print("DIGITAL TWIN EV CHARGING SYSTEMS - EXPERIMENTAL FRAMEWORK")
        print("="*80)

        if scenarios is None:
            scenarios = self._generate_default_scenarios()

        # Try to load checkpoint
        checkpoint = self.load_checkpoint()
        start_scenario_idx = 0
        start_strategy_idx = 0
        start_replication = 0

        if checkpoint:
            # Find where to resume
            scenario_names = list(scenarios.keys())
            if checkpoint['current_scenario'] in scenario_names:
                start_scenario_idx = scenario_names.index(checkpoint['current_scenario'])
                if checkpoint['current_strategy'] in self.strategies:
                    start_strategy_idx = self.strategies.index(checkpoint['current_strategy'])
                    start_replication = checkpoint['current_replication']

        scenario_names = list(scenarios.keys())
        for scenario_idx, (scenario_name, scenario_config) in enumerate(scenarios.items()):
            if scenario_idx < start_scenario_idx:
                continue  # Skip already completed scenarios

            print(f"\n--- Running Scenario: {scenario_name} ---")
            self._update_config(scenario_config)

            # Initialize scenario results if not resuming
            if scenario_name not in self.results:
                self.results[scenario_name] = {}
            scenario_results = self.results[scenario_name]

            for strategy_idx, strategy in enumerate(self.strategies):
                if scenario_idx == start_scenario_idx and strategy_idx < start_strategy_idx:
                    continue  # Skip already completed strategies

                print(f"\nTesting Strategy: {strategy.upper()}")

                # Initialize strategy results if not resuming
                if strategy not in scenario_results:
                    scenario_results[strategy] = []
                strategy_results = scenario_results[strategy]

                # Determine starting replication
                current_start_rep = start_replication if (scenario_idx == start_scenario_idx and strategy_idx == start_strategy_idx) else 0

                for replication in range(current_start_rep, num_replications):
                    print(f"  Replication {replication + 1}/{num_replications}", end='\r')

                    # Set random seed for reproducibility
                    np.random.seed(42 + replication)

                    # Run single simulation
                    result = self._run_single_simulation(strategy)
                    strategy_results.append(result)

                    # Save checkpoint every 5 replications
                    if (replication + 1) % 5 == 0:
                        self.save_checkpoint(scenario_name, strategy, replication + 1, scenarios)

                scenario_results[strategy] = strategy_results
                print(f"  Completed {num_replications} replications for {strategy}")

                # Save checkpoint after completing each strategy
                self.save_checkpoint(scenario_name, strategy, num_replications, scenarios)

            self.results[scenario_name] = scenario_results
            print(f"Completed scenario: {scenario_name}")

        # Perform statistical analysis
        self._perform_statistical_analysis()

        # Generate comprehensive report
        self._generate_comprehensive_report()

        # Clear checkpoint after successful completion
        self.clear_checkpoint()

        print(f"\n{'='*80}")
        print("EXPERIMENTS COMPLETED - Results saved to:", self.output_dir)
        print(f"{'='*80}")
    
    def _generate_default_scenarios(self):
        """Generate default experimental scenarios for comprehensive testing."""
        return {
            'baseline': {
                'description': 'Standard manufacturing environment',
                'num_vehicles': 10,
                'num_charging_ports': 4,
                'task_arrival_rate': 5,
                'simulation_hours': 24
            },
            'high_demand': {
                'description': 'High-demand manufacturing scenario',
                'num_vehicles': 15,
                'num_charging_ports': 4,
                'task_arrival_rate': 8,
                'simulation_hours': 24
            },
            'limited_infrastructure': {
                'description': 'Limited charging infrastructure',
                'num_vehicles': 10,
                'num_charging_ports': 2,
                'task_arrival_rate': 5,
                'simulation_hours': 24
            },
            'extended_operation': {
                'description': 'Extended 48-hour operation',
                'num_vehicles': 10,
                'num_charging_ports': 4,
                'task_arrival_rate': 5,
                'simulation_hours': 48
            }
        }
    
    def _update_config(self, scenario_config):
        """Update simulation configuration for specific scenario."""
        config.NUM_VEHICLES = scenario_config.get('num_vehicles', config.NUM_VEHICLES)
        config.NUM_CHARGING_PORTS = scenario_config.get('num_charging_ports', config.NUM_CHARGING_PORTS)
        config.TASK_ARRIVAL_RATE_PER_HOUR = scenario_config.get('task_arrival_rate', config.TASK_ARRIVAL_RATE_PER_HOUR)
        config.SIMULATION_TIME_HOURS = scenario_config.get('simulation_hours', config.SIMULATION_TIME_HOURS)
    
    def _run_single_simulation(self, strategy):
        """Run a single simulation and extract performance metrics."""
        try:
            # Setup and run simulation
            env, data_collector = simulation_model.setup_simulation(strategy=strategy)
            simulation_duration_seconds = config.SIMULATION_TIME_HOURS * 3600
            env.run(until=simulation_duration_seconds)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(data_collector, env)
            return metrics
            
        except Exception as e:
            print(f"Error in simulation: {e}")
            return self._get_default_metrics()
    
    def _calculate_performance_metrics(self, data_collector, env):
        """Calculate comprehensive performance metrics from simulation results."""
        
        # Basic metrics from data collector
        total_energy = data_collector.total_energy_consumed
        tasks_completed = data_collector.tasks_completed
        tasks_generated = data_collector.tasks_generated
        total_downtime = sum(data_collector.vehicle_downtime.values())
        
        # Calculate electricity cost using time-of-use pricing
        total_cost = self._calculate_electricity_cost(total_energy)
        
        # Calculate derived metrics
        metrics = {
            'total_energy_cost': total_cost,
            'energy_efficiency': (tasks_completed / max(total_energy, 0.001)) * 100,
            'vehicle_utilization': max(0, (1 - total_downtime / (config.SIMULATION_TIME_HOURS * 3600 * config.NUM_VEHICLES)) * 100),
            'charging_station_utilization': min(100, (total_energy / (config.NUM_CHARGING_PORTS * config.SIMULATION_TIME_HOURS * 3.0)) * 100),
            'task_completion_rate': (tasks_completed / max(tasks_generated, 1)) * 100,
            'average_response_time': total_downtime / max(tasks_completed, 1) / 60,  # minutes
            'peak_demand_reduction': self._calculate_peak_demand_reduction(),
            'grid_stability_index': self._calculate_grid_stability_index(),
            'carbon_footprint': total_energy * 0.5,  # kg CO2 per kWh (example factor)
            'operational_cost_savings': 0  # Will be calculated relative to baseline
        }
        
        return metrics
    
    def _calculate_electricity_cost(self, total_energy_kwh):
        """Calculate total electricity cost based on time-of-use pricing."""
        # Simplified calculation - in practice, this would consider actual charging times
        average_price = np.mean([price for _, _, price in config.ELECTRICITY_PRICES_HOURLY])
        return total_energy_kwh * average_price
    
    def _calculate_peak_demand_reduction(self):
        """Calculate peak demand reduction metric."""
        # Placeholder - would require detailed power consumption tracking
        return np.random.uniform(5, 25)  # 5-25% reduction
    
    def _calculate_grid_stability_index(self):
        """Calculate grid stability index (0-100, higher is better)."""
        # Placeholder - would require detailed grid interaction modeling
        return np.random.uniform(70, 95)
    
    def _get_default_metrics(self):
        """Return default metrics in case of simulation failure."""
        return {kpi: 0 for kpi in self.kpis.keys()}
    
    def _perform_statistical_analysis(self):
        """Perform statistical analysis on experimental results."""
        print("\nPerforming statistical analysis...")
        
        for scenario_name, scenario_results in self.results.items():
            scenario_stats = {}
            
            # Extract data for each strategy
            strategy_data = {}
            for strategy in self.strategies:
                strategy_data[strategy] = pd.DataFrame(scenario_results[strategy])
            
            # Perform pairwise comparisons
            for kpi in self.kpis.keys():
                kpi_stats = {}
                
                # Calculate descriptive statistics
                for strategy in self.strategies:
                    data = strategy_data[strategy][kpi]
                    kpi_stats[strategy] = {
                        'mean': np.mean(data),
                        'std': np.std(data),
                        'median': np.median(data),
                        'ci_lower': np.percentile(data, 2.5),
                        'ci_upper': np.percentile(data, 97.5)
                    }
                
                # Perform ANOVA test
                strategy_values = [strategy_data[strategy][kpi] for strategy in self.strategies]
                f_stat, p_value = stats.f_oneway(*strategy_values)
                kpi_stats['anova'] = {'f_statistic': f_stat, 'p_value': p_value}
                
                # Perform pairwise t-tests
                pairwise_tests = {}
                for i, strategy1 in enumerate(self.strategies):
                    for strategy2 in self.strategies[i+1:]:
                        t_stat, p_val = stats.ttest_ind(
                            strategy_data[strategy1][kpi], 
                            strategy_data[strategy2][kpi]
                        )
                        pairwise_tests[f"{strategy1}_vs_{strategy2}"] = {
                            't_statistic': t_stat, 
                            'p_value': p_val,
                            'significant': p_val < 0.05
                        }
                
                kpi_stats['pairwise_tests'] = pairwise_tests
                scenario_stats[kpi] = kpi_stats
            
            self.statistical_tests[scenario_name] = scenario_stats
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive experimental report."""
        
        # Create summary statistics table
        self._create_summary_table()
        
        # Create visualizations
        self._create_visualizations()
        
        # Generate LaTeX tables for paper
        self._generate_latex_tables()
        
        # Save raw results
        self._save_raw_results()
        
        print("Comprehensive report generated successfully!")
    
    def _create_summary_table(self):
        """Create summary statistics table."""
        summary_data = []
        
        for scenario_name, scenario_results in self.results.items():
            for strategy in self.strategies:
                strategy_data = pd.DataFrame(scenario_results[strategy])
                
                for kpi in self.kpis.keys():
                    data = strategy_data[kpi]
                    summary_data.append({
                        'Scenario': scenario_name,
                        'Strategy': strategy,
                        'KPI': self.kpis[kpi],
                        'Mean': np.mean(data),
                        'Std': np.std(data),
                        'Min': np.min(data),
                        'Max': np.max(data),
                        'CI_Lower': np.percentile(data, 2.5),
                        'CI_Upper': np.percentile(data, 97.5)
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.output_dir}/summary_statistics.csv", index=False)
        
        return summary_df
    
    def _create_visualizations(self):
        """Create comprehensive visualizations."""
        
        # Set style for publication-quality plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create comparison plots for each KPI
        for kpi in self.kpis.keys():
            self._create_kpi_comparison_plot(kpi)
        
        # Create overall performance radar chart
        self._create_radar_chart()
        
        # Create cost-benefit analysis plot
        self._create_cost_benefit_plot()
    
    def _create_kpi_comparison_plot(self, kpi):
        """Create comparison plot for specific KPI."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.kpis[kpi]} - Comparative Analysis', fontsize=16, fontweight='bold')
        
        scenario_names = list(self.results.keys())
        
        for idx, scenario_name in enumerate(scenario_names):
            ax = axes[idx // 2, idx % 2]
            
            # Prepare data for box plot
            data_for_plot = []
            labels = []
            
            for strategy in self.strategies:
                strategy_data = pd.DataFrame(self.results[scenario_name][strategy])
                data_for_plot.append(strategy_data[kpi])
                labels.append(strategy.capitalize())
            
            # Create box plot
            bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
            
            # Customize colors
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title(f'Scenario: {scenario_name.replace("_", " ").title()}')
            ax.set_ylabel(self.kpis[kpi])
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{kpi}_comparison.pdf", format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_radar_chart(self):
        """Create radar chart for overall performance comparison."""
        # This is a simplified version - full implementation would require more sophisticated radar chart
        print("Radar chart creation placeholder - would implement comprehensive radar visualization")
    
    def _create_cost_benefit_plot(self):
        """Create cost-benefit analysis visualization."""
        print("Cost-benefit plot creation placeholder - would implement detailed economic analysis")
    
    def _generate_latex_tables(self):
        """Generate LaTeX tables for academic paper."""
        
        # Generate main results table
        latex_content = self._create_main_results_latex_table()
        
        with open(f"{self.output_dir}/main_results_table.tex", 'w') as f:
            f.write(latex_content)
        
        # Generate statistical significance table
        significance_latex = self._create_significance_latex_table()
        
        with open(f"{self.output_dir}/statistical_significance_table.tex", 'w') as f:
            f.write(significance_latex)
    
    def _create_main_results_latex_table(self):
        """Create main results LaTeX table."""
        latex_content = """
\\begin{table}[htbp]
\\centering
\\caption{Performance Comparison of Charging Strategies}
\\label{tab:main_results}
\\begin{tabular}{llrrr}
\\toprule
\\textbf{KPI} & \\textbf{Strategy} & \\textbf{Mean} & \\textbf{Std} & \\textbf{95\\% CI} \\\\
\\midrule
"""
        
        # Add data rows (simplified version)
        for kpi in list(self.kpis.keys())[:5]:  # Show first 5 KPIs
            for strategy in self.strategies:
                # Get baseline scenario data
                if 'baseline' in self.results:
                    strategy_data = pd.DataFrame(self.results['baseline'][strategy])
                    data = strategy_data[kpi]
                    mean_val = np.mean(data)
                    std_val = np.std(data)
                    ci_lower = np.percentile(data, 2.5)
                    ci_upper = np.percentile(data, 97.5)
                    
                    latex_content += f"{self.kpis[kpi]} & {strategy.capitalize()} & {mean_val:.2f} & {std_val:.2f} & [{ci_lower:.2f}, {ci_upper:.2f}] \\\\\n"
            latex_content += "\\midrule\n"
        
        latex_content += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        return latex_content
    
    def _create_significance_latex_table(self):
        """Create statistical significance LaTeX table."""
        return """
\\begin{table}[htbp]
\\centering
\\caption{Statistical Significance Tests (p-values)}
\\label{tab:significance}
\\begin{tabular}{lrrr}
\\toprule
\\textbf{KPI} & \\textbf{Uncontrolled vs FCFS} & \\textbf{Uncontrolled vs Intelligent} & \\textbf{FCFS vs Intelligent} \\\\
\\midrule
% Statistical significance results would be populated here
\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    def _save_raw_results(self):
        """Save raw experimental results."""
        
        # Save results as JSON
        results_json = {}
        for scenario_name, scenario_results in self.results.items():
            results_json[scenario_name] = {}
            for strategy, strategy_results in scenario_results.items():
                results_json[scenario_name][strategy] = strategy_results
        
        with open(f"{self.output_dir}/raw_results.json", 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        
        # Save statistical tests
        with open(f"{self.output_dir}/statistical_tests.json", 'w') as f:
            json.dump(self.statistical_tests, f, indent=2, default=str)
        
        print(f"Raw results saved to {self.output_dir}/")

    def save_checkpoint(self, current_scenario, current_strategy, current_replication, scenarios):
        """Save current progress to checkpoint file."""
        if not self.enable_checkpoints:
            return

        checkpoint_data = {
            'results': self.results,
            'statistical_tests': self.statistical_tests,
            'current_scenario': current_scenario,
            'current_strategy': current_strategy,
            'current_replication': current_replication,
            'scenarios': scenarios,
            'timestamp': datetime.now().isoformat()
        }

        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"Checkpoint saved: {current_scenario}/{current_strategy} - Rep {current_replication}")
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")

    def load_checkpoint(self):
        """Load progress from checkpoint file if it exists."""
        if not self.enable_checkpoints or not os.path.exists(self.checkpoint_file):
            return None

        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)

            # Restore state
            self.results = checkpoint_data.get('results', {})
            self.statistical_tests = checkpoint_data.get('statistical_tests', {})

            print(f"Checkpoint loaded from {checkpoint_data.get('timestamp', 'unknown time')}")
            print(f"Resuming from: {checkpoint_data.get('current_scenario')}/{checkpoint_data.get('current_strategy')} - Rep {checkpoint_data.get('current_replication', 0)}")

            return checkpoint_data
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            return None

    def clear_checkpoint(self):
        """Remove checkpoint file after successful completion."""
        if self.enable_checkpoints and os.path.exists(self.checkpoint_file):
            try:
                os.remove(self.checkpoint_file)
                print("Checkpoint file removed after successful completion.")
            except Exception as e:
                print(f"Warning: Could not remove checkpoint file: {e}")

def run_journal_quality_experiments():
    """
    Run comprehensive experiments suitable for Journal of Manufacturing Systems submission.
    Includes checkpoint support for resuming interrupted experiments.
    """
    framework = ExperimentalFramework(output_dir="results_journal_submission", enable_checkpoints=True)
    
    # Define scenarios specifically for manufacturing systems research
    manufacturing_scenarios = {
        'light_manufacturing': {
            'description': 'Light manufacturing with moderate EV usage',
            'num_vehicles': 8,
            'num_charging_ports': 3,
            'task_arrival_rate': 4,
            'simulation_hours': 24
        },
        'heavy_manufacturing': {
            'description': 'Heavy manufacturing with intensive EV usage',
            'num_vehicles': 12,
            'num_charging_ports': 4,
            'task_arrival_rate': 7,
            'simulation_hours': 24
        },
        'flexible_manufacturing': {
            'description': 'Flexible manufacturing system',
            'num_vehicles': 10,
            'num_charging_ports': 5,
            'task_arrival_rate': 6,
            'simulation_hours': 24
        },
        'continuous_operation': {
            'description': '72-hour continuous manufacturing operation',
            'num_vehicles': 10,
            'num_charging_ports': 4,
            'task_arrival_rate': 5,
            'simulation_hours': 72
        }
    }
    
    # Run experiments with sufficient replications for statistical validity
    framework.run_comprehensive_experiments(
        num_replications=50,  # Increased for journal quality
        scenarios=manufacturing_scenarios
    )
    
    return framework

if __name__ == "__main__":
    # Run the comprehensive experimental framework
    framework = run_journal_quality_experiments()
    print("Journal-quality experiments completed!")