"""
Comprehensive Sensitivity Analysis for Digital Twin EV Charging Systems
======================================================================

Validates system robustness and operational boundaries through comprehensive
sensitivity analysis across key parameters.

Experiments include:
8.1 Energy modeling parameter sensitivity:
    - Charging efficiency variations (±10% from nominal 92%)
    - Vehicle energy consumption variations (±15% from nominal values)
8.2 Economic parameter sensitivity:
    - Objective weight perturbations (±20% from optimal values)
    - Energy tariff fluctuations (±25% from current rates)
    - Peak demand charge variations (±50% from baseline)
8.3 Infrastructure and operational sensitivity:
    - Fleet size variations (±30% from 10 vehicles)
    - Charging port availability reductions (from 4 to 2 ports)
    - Simultaneous multi-port failures (up to 50% capacity loss)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os
from datetime import datetime
import itertools
import sys
sys.path.append('..')
from .. import config
from .. import simulation_model

class SensitivityAnalysisExperiments:
    """
    Comprehensive sensitivity analysis for digital twin robustness validation.
    """
    
    def __init__(self, output_dir="results_sensitivity_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define sensitivity analysis parameters for comprehensive evaluation
        self.sensitivity_parameters = {
            # 8.1 Energy modeling parameters
            'energy_parameters': {
                'charging_efficiency': {
                    'baseline': 0.92,
                    'variations': [-0.10, -0.05, 0.0, 0.05, 0.10],  # ±10%
                    'description': 'Charging efficiency variations'
                },
                'energy_consumption': {
                    'baseline': 1.0,
                    'variations': [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15],  # ±15%
                    'description': 'Vehicle energy consumption variations'
                }
            },
            # 8.2 Economic parameters
            'economic_parameters': {
                'objective_weights': {
                    'baseline': [0.4, 0.4, 0.2],  # α₁, α₂, α₃
                    'variations': [-0.20, -0.10, 0.0, 0.10, 0.20],  # ±20%
                    'description': 'Multi-objective weight perturbations'
                },
                'energy_tariffs': {
                    'baseline': 1.0,
                    'variations': [-0.25, -0.15, -0.05, 0.0, 0.05, 0.15, 0.25],  # ±25%
                    'description': 'Energy tariff fluctuations'
                },
                'peak_demand_charges': {
                    'baseline': 1.0,
                    'variations': [-0.50, -0.25, 0.0, 0.25, 0.50],  # ±50%
                    'description': 'Peak demand charge variations'
                }
            },
            # 8.3 Infrastructure and operational parameters
            'infrastructure_parameters': {
                'fleet_size': {
                    'baseline': 10,
                    'variations': [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30],  # ±30%
                    'description': 'Fleet size variations'
                },
                'charging_ports': {
                    'baseline': 4,
                    'variations': [2, 3, 4, 5, 6],  # Absolute values
                    'description': 'Charging port availability'
                },
                'port_failures': {
                    'baseline': 0.0,
                    'variations': [0.0, 0.125, 0.25, 0.375, 0.50],  # 0-50% failures
                    'description': 'Simultaneous port failure rates'
                }
            }
        }
        
        self.results = {}
        
    def run_comprehensive_sensitivity_analysis(self, num_replications=20):
        """
        Run comprehensive sensitivity analysis experiments.
        
        Args:
            num_replications (int): Number of replications for statistical validity
        """
        print("="*80)
        print("COMPREHENSIVE SENSITIVITY ANALYSIS - DIGITAL TWIN ROBUSTNESS")
        print("="*80)
        
        # Run individual parameter sensitivity analyses
        for category_name, category_params in self.sensitivity_parameters.items():
            print(f"\n--- {category_name.upper()} SENSITIVITY ANALYSIS ---")
            
            category_results = {}
            
            for param_name, param_config in category_params.items():
                print(f"\nAnalyzing parameter: {param_name}")
                print(f"Description: {param_config['description']}")
                
                param_results = []
                
                for variation in param_config['variations']:
                    print(f"  Testing variation: {variation:+.1%}" if isinstance(variation, float) and abs(variation) < 1 
                          else f"  Testing value: {variation}")
                    
                    variation_results = []
                    
                    for replication in range(num_replications):
                        # Set random seed for reproducibility
                        np.random.seed(42 + replication)
                        
                        # Run simulation with parameter variation
                        result = self._run_sensitivity_simulation(
                            category_name, param_name, param_config, variation
                        )
                        variation_results.append(result)
                    
                    param_results.append({
                        'variation': variation,
                        'results': variation_results
                    })
                
                category_results[param_name] = param_results
            
            self.results[category_name] = category_results
        
        # Run combined sensitivity analysis (interaction effects)
        self._run_combined_sensitivity_analysis(num_replications)
        
        # Analyze results and generate reports
        self._analyze_sensitivity_results()
        self._generate_sensitivity_analysis_report()
        
        print(f"\n{'='*80}")
        print("SENSITIVITY ANALYSIS COMPLETED - Results saved to:", self.output_dir)
        print(f"{'='*80}")
    
    def _run_sensitivity_simulation(self, category_name, param_name, param_config, variation):
        """
        Run simulation with specific parameter variation.
        
        Args:
            category_name (str): Parameter category
            param_name (str): Parameter name
            param_config (dict): Parameter configuration
            variation (float): Parameter variation
            
        Returns:
            dict: Performance metrics under parameter variation
        """
        # Create modified configuration
        modified_config = self._create_modified_config(
            category_name, param_name, param_config, variation
        )
        
        # Run simulation with modified parameters
        try:
            env, data_collector = simulation_model.setup_simulation(
                strategy='intelligent',
                modified_config=modified_config
            )
            
            # Run simulation
            simulation_duration = config.SIMULATION_TIME_HOURS * 3600
            env.run(until=simulation_duration)
            
            # Extract performance metrics
            metrics = self._extract_performance_metrics(data_collector, modified_config)
            
            # Add sensitivity-specific metrics
            sensitivity_metrics = self._calculate_sensitivity_metrics(
                metrics, category_name, param_name, variation
            )
            metrics.update(sensitivity_metrics)
            
            return metrics
            
        except Exception as e:
            print(f"    Error in simulation: {e}")
            return self._get_default_metrics()
    
    def _create_modified_config(self, category_name, param_name, param_config, variation):
        """
        Create modified configuration based on parameter variation.
        
        Args:
            category_name (str): Parameter category
            param_name (str): Parameter name
            param_config (dict): Parameter configuration
            variation (float): Parameter variation
            
        Returns:
            dict: Modified configuration
        """
        modified_config = {
            'category': category_name,
            'parameter': param_name,
            'variation': variation,
            'baseline': param_config['baseline']
        }
        
        if category_name == 'energy_parameters':
            if param_name == 'charging_efficiency':
                modified_config['charging_efficiency'] = param_config['baseline'] * (1 + variation)
            elif param_name == 'energy_consumption':
                modified_config['energy_consumption_multiplier'] = 1 + variation
                
        elif category_name == 'economic_parameters':
            if param_name == 'objective_weights':
                baseline_weights = param_config['baseline']
                # Apply variation to each weight while maintaining sum = 1
                modified_weights = []
                for i, base_weight in enumerate(baseline_weights):
                    varied_weight = base_weight * (1 + variation)
                    modified_weights.append(varied_weight)
                
                # Normalize to sum = 1
                weight_sum = sum(modified_weights)
                modified_config['objective_weights'] = [w / weight_sum for w in modified_weights]
                
            elif param_name == 'energy_tariffs':
                modified_config['tariff_multiplier'] = 1 + variation
                
            elif param_name == 'peak_demand_charges':
                modified_config['peak_demand_multiplier'] = 1 + variation
                
        elif category_name == 'infrastructure_parameters':
            if param_name == 'fleet_size':
                if isinstance(variation, float):
                    modified_config['num_vehicles'] = max(1, int(param_config['baseline'] * (1 + variation)))
                else:
                    modified_config['num_vehicles'] = variation
                    
            elif param_name == 'charging_ports':
                modified_config['num_charging_ports'] = variation
                
            elif param_name == 'port_failures':
                modified_config['port_failure_rate'] = variation
        
        return modified_config
    
    def _extract_performance_metrics(self, data_collector, modified_config):
        """Extract performance metrics from simulation."""
        total_energy = data_collector.total_energy_consumed
        tasks_completed = data_collector.tasks_completed
        tasks_generated = data_collector.tasks_generated
        total_downtime = sum(data_collector.vehicle_downtime.values())
        
        # Calculate derived metrics
        num_vehicles = modified_config.get('num_vehicles', config.NUM_VEHICLES)
        simulation_hours = config.SIMULATION_TIME_HOURS
        
        return {
            'total_energy_consumed': total_energy,
            'total_energy_charged': data_collector.total_energy_charged,
            'tasks_completed': tasks_completed,
            'tasks_generated': tasks_generated,
            'task_completion_rate': (tasks_completed / max(tasks_generated, 1)) * 100,
            'total_downtime_hours': total_downtime / 3600,
            'vehicle_utilization': max(0, (1 - total_downtime / (simulation_hours * 3600 * num_vehicles)) * 100),
            'energy_efficiency': (tasks_completed / max(total_energy, 0.001)) * 100,
            'charging_efficiency': data_collector.total_energy_charged / max(data_collector.total_energy_consumed + data_collector.total_energy_charged, 0.001),
            'operational_cost': self._calculate_operational_cost(data_collector, modified_config)
        }
    
    def _calculate_operational_cost(self, data_collector, modified_config):
        """Calculate operational cost with parameter variations."""
        # Base energy cost
        energy_cost = data_collector.total_energy_charged * 0.15  # Base rate
        
        # Apply tariff multiplier if applicable
        if 'tariff_multiplier' in modified_config:
            energy_cost *= modified_config['tariff_multiplier']
        
        # Add peak demand charges if applicable
        peak_demand_cost = 0
        if 'peak_demand_multiplier' in modified_config:
            peak_demand_cost = 50 * modified_config['peak_demand_multiplier']  # Base peak charge
        
        # Add downtime costs
        downtime_cost = sum(data_collector.vehicle_downtime.values()) / 3600 * 100  # $100/hour
        
        return energy_cost + peak_demand_cost + downtime_cost
    
    def _calculate_sensitivity_metrics(self, metrics, category_name, param_name, variation):
        """Calculate sensitivity-specific metrics."""
        return {
            'parameter_category': category_name,
            'parameter_name': param_name,
            'parameter_variation': variation,
            'sensitivity_index': abs(variation) if isinstance(variation, float) else 0
        }
    
    def _get_default_metrics(self):
        """Return default metrics in case of simulation failure."""
        return {
            'total_energy_consumed': 0,
            'total_energy_charged': 0,
            'tasks_completed': 0,
            'tasks_generated': 1,
            'task_completion_rate': 0,
            'total_downtime_hours': 24,
            'vehicle_utilization': 0,
            'energy_efficiency': 0,
            'charging_efficiency': 0,
            'operational_cost': 1000,
            'parameter_category': 'unknown',
            'parameter_name': 'unknown',
            'parameter_variation': 0,
            'sensitivity_index': 0
        }
    
    def _run_combined_sensitivity_analysis(self, num_replications):
        """
        Run combined sensitivity analysis to test interaction effects.
        
        Args:
            num_replications (int): Number of replications
        """
        print("\n--- COMBINED SENSITIVITY ANALYSIS (INTERACTION EFFECTS) ---")
        
        # Test key parameter combinations
        combined_scenarios = [
            {
                'name': 'high_demand_low_infrastructure',
                'modifications': {
                    'energy_consumption_multiplier': 1.15,  # +15% consumption
                    'num_vehicles': 13,  # +30% fleet size
                    'num_charging_ports': 2,  # 50% port reduction
                },
                'description': 'High demand with limited infrastructure'
            },
            {
                'name': 'economic_stress_test',
                'modifications': {
                    'tariff_multiplier': 1.25,  # +25% tariffs
                    'peak_demand_multiplier': 1.50,  # +50% peak charges
                    'objective_weights': [0.32, 0.32, 0.16],  # -20% weight variation
                },
                'description': 'Economic stress test scenario'
            },
            {
                'name': 'worst_case_scenario',
                'modifications': {
                    'charging_efficiency': 0.828,  # -10% efficiency
                    'energy_consumption_multiplier': 1.15,  # +15% consumption
                    'port_failure_rate': 0.50,  # 50% port failures
                    'tariff_multiplier': 1.25,  # +25% tariffs
                },
                'description': 'Worst-case combined scenario'
            }
        ]
        
        combined_results = {}
        
        for scenario in combined_scenarios:
            print(f"\nTesting combined scenario: {scenario['name']}")
            print(f"Description: {scenario['description']}")
            
            scenario_results = []
            
            for replication in range(num_replications):
                np.random.seed(42 + replication)
                
                # Create combined modified config
                modified_config = scenario['modifications'].copy()
                modified_config['scenario_name'] = scenario['name']
                
                try:
                    env, data_collector = simulation_model.setup_simulation(
                        strategy='intelligent',
                        modified_config=modified_config
                    )
                    
                    simulation_duration = config.SIMULATION_TIME_HOURS * 3600
                    env.run(until=simulation_duration)
                    
                    metrics = self._extract_performance_metrics(data_collector, modified_config)
                    metrics['scenario_name'] = scenario['name']
                    scenario_results.append(metrics)
                    
                except Exception as e:
                    print(f"    Error in combined simulation: {e}")
                    default_metrics = self._get_default_metrics()
                    default_metrics['scenario_name'] = scenario['name']
                    scenario_results.append(default_metrics)
            
            combined_results[scenario['name']] = scenario_results
        
        self.results['combined_scenarios'] = combined_results

    def _analyze_sensitivity_results(self):
        """
        Analyze sensitivity analysis results and validate against performance targets.
        """
        print("\nAnalyzing sensitivity analysis results...")

        # Expected performance targets:
        # - Energy parameter variations: 2.1-3.4% cost changes, 1.8-2.9% performance changes
        # - Economic parameter variations: <5% performance degradation with ±20% weight changes
        # - Infrastructure variations: near-linear cost scaling, 97% task completion with port reductions

        expected_results = {
            'energy_param_cost_change_max': 3.4,
            'energy_param_performance_change_max': 2.9,
            'economic_param_performance_degradation_max': 5.0,
            'infrastructure_task_completion_min': 97.0
        }

        self.analysis_results = {}

        # Analyze individual parameter sensitivities
        for category_name, category_results in self.results.items():
            if category_name == 'combined_scenarios':
                continue  # Handle separately

            category_analysis = {}

            for param_name, param_results in category_results.items():
                param_analysis = self._analyze_parameter_sensitivity(
                    param_name, param_results, expected_results
                )
                category_analysis[param_name] = param_analysis

            self.analysis_results[category_name] = category_analysis

        # Analyze combined scenarios
        if 'combined_scenarios' in self.results:
            self.analysis_results['combined_scenarios'] = self._analyze_combined_scenarios(
                self.results['combined_scenarios'], expected_results
            )

    def _analyze_parameter_sensitivity(self, param_name, param_results, expected_results):
        """
        Analyze sensitivity for individual parameter.

        Args:
            param_name (str): Parameter name
            param_results (list): Results for parameter variations
            expected_results (dict): Expected validation criteria

        Returns:
            dict: Parameter sensitivity analysis
        """
        # Extract baseline (variation = 0) results
        baseline_results = None
        for variation_data in param_results:
            if variation_data['variation'] == 0 or variation_data['variation'] == 0.0:
                baseline_results = variation_data['results']
                break

        if not baseline_results:
            # Use first variation as baseline if no zero variation
            baseline_results = param_results[0]['results']

        baseline_metrics = pd.DataFrame(baseline_results).mean()

        # Calculate sensitivity metrics for each variation
        sensitivity_data = []

        for variation_data in param_results:
            variation = variation_data['variation']
            variation_results = variation_data['results']

            if not variation_results:
                continue

            variation_metrics = pd.DataFrame(variation_results).mean()

            # Calculate percentage changes from baseline
            cost_change = ((variation_metrics['operational_cost'] - baseline_metrics['operational_cost']) /
                          baseline_metrics['operational_cost'] * 100) if baseline_metrics['operational_cost'] > 0 else 0

            performance_change = ((baseline_metrics['task_completion_rate'] - variation_metrics['task_completion_rate']) /
                                baseline_metrics['task_completion_rate'] * 100) if baseline_metrics['task_completion_rate'] > 0 else 0

            efficiency_change = ((baseline_metrics['energy_efficiency'] - variation_metrics['energy_efficiency']) /
                               baseline_metrics['energy_efficiency'] * 100) if baseline_metrics['energy_efficiency'] > 0 else 0

            sensitivity_data.append({
                'variation': variation,
                'cost_change_percent': cost_change,
                'performance_change_percent': performance_change,
                'efficiency_change_percent': efficiency_change,
                'task_completion_rate': variation_metrics['task_completion_rate'],
                'vehicle_utilization': variation_metrics['vehicle_utilization']
            })

        sensitivity_df = pd.DataFrame(sensitivity_data)

        # Validate against expected results
        validation_status = "PASS"
        validation_notes = []

        if param_name in ['charging_efficiency', 'energy_consumption']:
            max_cost_change = sensitivity_df['cost_change_percent'].abs().max()
            max_performance_change = sensitivity_df['performance_change_percent'].abs().max()

            if max_cost_change > expected_results['energy_param_cost_change_max']:
                validation_status = "FAIL"
                validation_notes.append(f"Cost change {max_cost_change:.1f}% exceeds limit")

            if max_performance_change > expected_results['energy_param_performance_change_max']:
                validation_status = "FAIL"
                validation_notes.append(f"Performance change {max_performance_change:.1f}% exceeds limit")

        elif param_name == 'objective_weights':
            max_performance_degradation = sensitivity_df['performance_change_percent'].max()

            if max_performance_degradation > expected_results['economic_param_performance_degradation_max']:
                validation_status = "FAIL"
                validation_notes.append(f"Performance degradation {max_performance_degradation:.1f}% exceeds limit")

        elif param_name == 'charging_ports':
            min_task_completion = sensitivity_df['task_completion_rate'].min()

            if min_task_completion < expected_results['infrastructure_task_completion_min']:
                validation_status = "FAIL"
                validation_notes.append(f"Task completion {min_task_completion:.1f}% below minimum")

        return {
            'sensitivity_data': sensitivity_df,
            'max_cost_change': sensitivity_df['cost_change_percent'].abs().max(),
            'max_performance_change': sensitivity_df['performance_change_percent'].abs().max(),
            'robustness_score': 1.0 - (sensitivity_df['performance_change_percent'].abs().mean() / 100.0),
            'validation_status': validation_status,
            'validation_notes': validation_notes
        }

    def _analyze_combined_scenarios(self, combined_results, expected_results):
        """
        Analyze combined scenario results.

        Args:
            combined_results (dict): Combined scenario results
            expected_results (dict): Expected validation criteria

        Returns:
            dict: Combined scenario analysis
        """
        scenario_analysis = {}

        for scenario_name, scenario_results in combined_results.items():
            if not scenario_results:
                continue

            metrics_df = pd.DataFrame(scenario_results)
            mean_metrics = metrics_df.mean()

            # Calculate performance retention under stress
            baseline_performance = 95.0  # Assumed baseline performance
            performance_retention = (mean_metrics['task_completion_rate'] / baseline_performance * 100) if baseline_performance > 0 else 0

            validation_status = "PASS"
            if scenario_name == 'worst_case_scenario':
                # Worst case should retain at least 89% performance (as claimed)
                if performance_retention < 89.0:
                    validation_status = "FAIL"

            scenario_analysis[scenario_name] = {
                'mean_metrics': mean_metrics,
                'performance_retention': performance_retention,
                'robustness_under_stress': mean_metrics['vehicle_utilization'] / 100.0,
                'validation_status': validation_status
            }

        return scenario_analysis

    def _generate_sensitivity_analysis_report(self):
        """
        Generate comprehensive sensitivity analysis report.
        """
        print("\nGenerating sensitivity analysis report...")

        # Create summary statistics table
        summary_data = []

        for category_name, category_analysis in self.analysis_results.items():
            if category_name == 'combined_scenarios':
                continue  # Handle separately

            for param_name, param_analysis in category_analysis.items():
                summary_data.append({
                    'Category': category_name.replace('_', ' ').title(),
                    'Parameter': param_name.replace('_', ' ').title(),
                    'Max Cost Change (%)': f"{param_analysis['max_cost_change']:.1f}",
                    'Max Performance Change (%)': f"{param_analysis['max_performance_change']:.1f}",
                    'Robustness Score': f"{param_analysis['robustness_score']:.3f}",
                    'Validation': param_analysis['validation_status']
                })

        # Add combined scenarios
        if 'combined_scenarios' in self.analysis_results:
            for scenario_name, scenario_analysis in self.analysis_results['combined_scenarios'].items():
                summary_data.append({
                    'Category': 'Combined Scenarios',
                    'Parameter': scenario_name.replace('_', ' ').title(),
                    'Max Cost Change (%)': 'N/A',
                    'Max Performance Change (%)': f"{100 - scenario_analysis['performance_retention']:.1f}",
                    'Robustness Score': f"{scenario_analysis['robustness_under_stress']:.3f}",
                    'Validation': scenario_analysis['validation_status']
                })

        summary_df = pd.DataFrame(summary_data)

        # Save summary table
        summary_df.to_csv(
            os.path.join(self.output_dir, 'sensitivity_analysis_summary.csv'),
            index=False
        )

        # Generate visualizations
        self._create_sensitivity_analysis_visualizations()

        # Save detailed results
        with open(os.path.join(self.output_dir, 'sensitivity_analysis_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"Summary table saved: {self.output_dir}/sensitivity_analysis_summary.csv")
        print(f"Detailed results: {self.output_dir}/sensitivity_analysis_results.json")

    def _create_sensitivity_analysis_visualizations(self):
        """
        Create visualizations for sensitivity analysis.
        """
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Create comprehensive sensitivity analysis plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Parameter sensitivity tornado chart
        self._create_tornado_chart(ax1)

        # Plot 2: Cost vs performance trade-offs
        self._create_cost_performance_plot(ax2)

        # Plot 3: Robustness scores by category
        self._create_robustness_plot(ax3)

        # Plot 4: Combined scenario performance
        self._create_combined_scenario_plot(ax4)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sensitivity_analysis.pdf'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved: {self.output_dir}/sensitivity_analysis.pdf")

    def _create_tornado_chart(self, ax):
        """Create tornado chart for parameter sensitivity."""
        # Extract sensitivity data for tornado chart
        param_names = []
        max_changes = []

        for category_name, category_analysis in self.analysis_results.items():
            if category_name == 'combined_scenarios':
                continue

            for param_name, param_analysis in category_analysis.items():
                param_names.append(f"{param_name}")
                max_changes.append(param_analysis['max_performance_change'])

        if param_names and max_changes:
            # Sort by sensitivity (highest first)
            sorted_data = sorted(zip(param_names, max_changes), key=lambda x: abs(x[1]), reverse=True)
            param_names, max_changes = zip(*sorted_data)

            y_pos = np.arange(len(param_names))
            bars = ax.barh(y_pos, max_changes, alpha=0.7)

            # Color bars by magnitude
            for bar, change in zip(bars, max_changes):
                if abs(change) > 3:
                    bar.set_color('red')
                elif abs(change) > 1:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')

            ax.set_yticks(y_pos)
            ax.set_yticklabels([name.replace('_', ' ').title() for name in param_names])
            ax.set_xlabel('Max Performance Change (%)')
            ax.set_title('Parameter Sensitivity Tornado Chart')
            ax.grid(True, alpha=0.3)

    def _create_cost_performance_plot(self, ax):
        """Create cost vs performance trade-off plot."""
        cost_changes = []
        performance_changes = []
        param_labels = []

        for category_name, category_analysis in self.analysis_results.items():
            if category_name == 'combined_scenarios':
                continue

            for param_name, param_analysis in category_analysis.items():
                cost_changes.append(param_analysis['max_cost_change'])
                performance_changes.append(param_analysis['max_performance_change'])
                param_labels.append(param_name.replace('_', ' ').title())

        if cost_changes and performance_changes:
            scatter = ax.scatter(cost_changes, performance_changes, s=100, alpha=0.7)

            ax.set_xlabel('Max Cost Change (%)')
            ax.set_ylabel('Max Performance Change (%)')
            ax.set_title('Cost vs Performance Trade-offs')
            ax.grid(True, alpha=0.3)

            # Add parameter labels
            for i, label in enumerate(param_labels):
                ax.annotate(label, (cost_changes[i], performance_changes[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

    def _create_robustness_plot(self, ax):
        """Create robustness scores by category plot."""
        categories = []
        robustness_scores = []

        for category_name, category_analysis in self.analysis_results.items():
            if category_name == 'combined_scenarios':
                continue

            category_robustness = []
            for param_name, param_analysis in category_analysis.items():
                category_robustness.append(param_analysis['robustness_score'])

            if category_robustness:
                categories.append(category_name.replace('_', ' ').title())
                robustness_scores.append(np.mean(category_robustness))

        if categories and robustness_scores:
            bars = ax.bar(categories, robustness_scores, alpha=0.7)

            # Color bars by robustness level
            for bar, score in zip(bars, robustness_scores):
                if score > 0.95:
                    bar.set_color('green')
                elif score > 0.90:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')

            ax.set_ylabel('Robustness Score')
            ax.set_title('System Robustness by Parameter Category')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

    def _create_combined_scenario_plot(self, ax):
        """Create combined scenario performance plot."""
        if 'combined_scenarios' not in self.analysis_results:
            ax.text(0.5, 0.5, 'No combined scenario data', ha='center', va='center', transform=ax.transAxes)
            return

        scenarios = []
        performance_retentions = []

        for scenario_name, scenario_analysis in self.analysis_results['combined_scenarios'].items():
            scenarios.append(scenario_name.replace('_', ' ').title())
            performance_retentions.append(scenario_analysis['performance_retention'])

        if scenarios and performance_retentions:
            bars = ax.bar(scenarios, performance_retentions, alpha=0.7)

            # Add target line for worst-case scenario (89% retention)
            ax.axhline(y=89, color='red', linestyle='--', alpha=0.7, label='Target: 89%')

            # Color bars by performance level
            for bar, retention in zip(bars, performance_retentions):
                if retention > 95:
                    bar.set_color('green')
                elif retention > 89:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')

            ax.set_ylabel('Performance Retention (%)')
            ax.set_title('Combined Scenario Stress Test Results')
            ax.set_ylim(0, 100)
            ax.tick_params(axis='x', rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)

def main():
    """Run comprehensive sensitivity analysis experiments."""
    experiments = SensitivityAnalysisExperiments()
    experiments.run_comprehensive_sensitivity_analysis(num_replications=20)

if __name__ == "__main__":
    main()
