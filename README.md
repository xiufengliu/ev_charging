# Digital Twin EV Charging System

A comprehensive Digital Twin framework for intelligent electric vehicle charging optimization in smart manufacturing environments. This system integrates real-time simulation, machine learning, and multi-objective optimization to enable efficient and adaptive charging control for industrial vehicle fleets.

## Features

- **Digital Twin Architecture**: Real-time virtual representation of manufacturing systems and vehicle fleets
- **Intelligent Optimization**: Multi-objective optimization considering energy costs, vehicle availability, and operational constraints
- **Predictive Analytics**: Machine learning models for demand forecasting and system behavior prediction
- **Real-time Control**: Adaptive charging control with rolling horizon optimization
- **Crash Recovery**: Robust checkpoint system for long-running experiments
- **Comprehensive Simulation**: Discrete-event simulation of manufacturing environments and vehicle operations

## System Architecture

The framework consists of four main layers:

1. **Physical Layer**: Manufacturing environment, vehicle fleet, and charging infrastructure
2. **Data Integration Layer**: Real-time data processing and synchronization
3. **Digital Twin Layer**: Simulation models, ML algorithms, and optimization engines
4. **Control Layer**: Charging decision implementation and performance monitoring

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy simpy scikit-learn
```

### Clone Repository

```bash
git clone https://github.com/yourusername/ev-charging-digital-twin.git
cd ev-charging-digital-twin
```

## Quick Start

### Basic Simulation

Run a basic simulation with default parameters:

```bash
python run_experiments.py
```

### Custom Configuration

Modify simulation parameters in `simulation/config.py`:

```python
# Vehicle fleet configuration
NUM_VEHICLES = 10
BATTERY_CAPACITY = 5.0  # kWh
MAX_CHARGING_POWER = 3.3  # kW

# Simulation parameters
SIMULATION_TIME = 24  # hours
RANDOM_SEED = 42
```

### Advanced Usage

For custom experiments with specific scenarios:

```python
from simulation.experimental_framework import ExperimentalFramework

# Initialize framework
framework = ExperimentalFramework(output_dir="custom_results")

# Run experiments
framework.run_comprehensive_experiments(
    num_replications=30,
    scenarios=custom_scenarios
)
```

## Configuration

### Charging Strategies

The system supports multiple charging strategies:

- **Uncontrolled**: Basic threshold-based charging
- **FCFS**: First-come-first-served allocation
- **Intelligent**: AI-optimized charging with predictive control

### Manufacturing Scenarios

Pre-configured scenarios include:

- **Light Manufacturing**: 8 vehicles, moderate intensity
- **Heavy Manufacturing**: 12 vehicles, high intensity  
- **Flexible Manufacturing**: 10 vehicles, variable patterns
- **Continuous Operation**: 24/7 operations, extended periods

### Optimization Parameters

Key optimization parameters can be adjusted:

```python
# Objective function weights
ALPHA_ENERGY = 0.4      # Energy cost weight
ALPHA_DOWNTIME = 0.4    # Vehicle downtime weight
ALPHA_UTILIZATION = 0.2 # Infrastructure utilization weight

# Constraints
MIN_SOC = 0.2          # Minimum state of charge
MAX_SOC = 1.0          # Maximum state of charge
CHARGING_EFFICIENCY = 0.92  # Charging efficiency
```

## Monitoring and Analysis

### Real-time Monitoring

Monitor simulation progress:

```bash
python monitor_simulation.py
```

### Results Analysis

Results are automatically saved in structured formats:

- **CSV files**: Statistical summaries and time series data
- **JSON files**: Detailed simulation results and configurations
- **PDF figures**: Publication-quality visualizations

### Performance Metrics

The system tracks comprehensive performance metrics:

- Energy consumption and costs
- Vehicle availability and uptime
- Charging infrastructure utilization
- Task completion rates
- System responsiveness

## Advanced Features

### Checkpoint System

The framework includes robust crash recovery:

- Automatic progress saving every 5 replications
- Resume from last checkpoint after interruption
- Progress tracking and time estimation

### Machine Learning Integration

Built-in ML capabilities:

- LSTM networks for temporal pattern recognition
- Random Forest for non-linear relationships
- Gaussian Process for uncertainty quantification
- Ensemble methods for robust predictions

### Optimization Algorithms

Advanced optimization techniques:

- Mixed-integer programming formulations
- Rolling horizon optimization
- Hierarchical decomposition strategies
- Multi-objective optimization with adaptive weights

## API Reference

### Core Classes

#### ExperimentalFramework

Main class for running experiments and managing simulations.

```python
framework = ExperimentalFramework(
    output_dir="results",
    enable_checkpoints=True
)
```

#### SimulationModel

Discrete-event simulation of manufacturing environment.

```python
model = SimulationModel(
    num_vehicles=10,
    simulation_time=24,
    random_seed=42
)
```

#### OptimizationController

Multi-objective optimization engine for charging decisions.

```python
controller = OptimizationController(
    prediction_horizon=4,
    optimization_interval=1
)
```

### Key Methods

- `run_comprehensive_experiments()`: Execute full experimental evaluation
- `run_single_simulation()`: Run individual simulation instance
- `save_checkpoint()`: Save current progress
- `load_checkpoint()`: Resume from saved state
- `generate_reports()`: Create analysis reports and visualizations

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/ev-charging-digital-twin.git
cd ev-charging-digital-twin

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions and support:

- Create an issue on GitHub
- Check the documentation in the `docs/` folder
- Review example configurations in `examples/`

## Acknowledgments

- Built with Python and SimPy discrete-event simulation framework
- Optimization powered by advanced mathematical programming techniques
- Machine learning integration using scikit-learn and TensorFlow
- Visualization capabilities provided by matplotlib and seaborn
