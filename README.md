# Digital Twin Framework for EV Charging Optimization in Smart Manufacturing

This repository contains the implementation and experimental validation code for a Digital Twin framework that optimizes electric vehicle charging in smart manufacturing environments. The system employs hierarchical decision-making, machine learning ensemble methods, and multi-objective optimization to achieve significant improvements in energy efficiency and operational performance.

## Research Contributions

- **Digital Twin Architecture**: Four-layer framework integrating physical systems, data processing, virtual modeling, and intelligent control
- **Multi-Objective Optimization**: Simultaneous optimization of energy costs, vehicle availability, and charging infrastructure utilization
- **Machine Learning Ensemble**: LSTM, Random Forest, and Gaussian Process models for robust demand prediction
- **Hierarchical Control**: Strategic, tactical, and operational decision layers with rolling horizon optimization
- **Experimental Validation**: Comprehensive analysis including sensor noise robustness, communication delays, and carbon emission optimization

## System Performance

Experimental results demonstrate:
- 23.7% reduction in energy costs
- 18.4% improvement in fleet availability
- 31.2% increase in charging station utilization
- 94.8% prediction accuracy with ensemble methods
- Robust performance under ±5% sensor noise conditions

## Installation and Setup

### Requirements

- Python 3.8+
- Required packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `simpy`, `scikit-learn`

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import simulation; print('Installation successful')"
```

## Experimental Reproduction

### Complete Experimental Suite

Run all experiments reported in the paper:

```bash
# Full experimental validation (30 replications per scenario)
python run_experimental_validation.py

# Quick validation (reduced replications for testing)
python run_experimental_validation.py --quick
```

### Individual Experiment Categories

```bash
# Sensor noise robustness analysis
python -m simulation.experiments.sensor_noise_experiments

# LSTM prediction accuracy validation
python -m simulation.experiments.lstm_prediction_experiments

# Communication delay impact assessment
python -m simulation.experiments.communication_experiments

# Carbon emission optimization analysis
python -m simulation.experiments.carbon_emission_experiments

# Sensitivity analysis across parameters
python -m simulation.experiments.sensitivity_analysis_experiments
```

### Core Simulation Framework

```bash
# Run basic comparison of charging strategies
python run_experiments.py

# Generate publication-quality visualizations
python generate_figures.py
```

## Configuration

### System Parameters

Key parameters are defined in `simulation/config.py`:

```python
# Fleet configuration
NUM_VEHICLES = 10
BATTERY_CAPACITY_KWH = 5.0
CHARGING_RATE_KW = 3.3

# Optimization weights
ALPHA_ENERGY = 0.4      # Energy cost minimization
ALPHA_DOWNTIME = 0.4    # Vehicle availability maximization
ALPHA_UTILIZATION = 0.2 # Infrastructure utilization

# Prediction parameters
PREDICTION_HORIZON_HOURS = 4
UPDATE_INTERVAL_MINUTES = 10
```

## Results and Analysis

### Output Structure

Experimental results are organized as follows:

```
results_comprehensive/
├── results_sensor_noise/          # Sensor noise robustness analysis
├── results_lstm_prediction/       # LSTM prediction accuracy validation
├── results_communication/         # Communication delay impact
├── results_carbon_emission/       # Carbon optimization results
├── results_sensitivity_analysis/  # Parameter sensitivity analysis
└── comprehensive_summary_report.json
```

### Key Performance Metrics

The framework evaluates:

- **Energy Efficiency**: Total energy consumption and cost optimization
- **Fleet Availability**: Vehicle uptime and task completion rates
- **Infrastructure Utilization**: Charging station usage optimization
- **Prediction Accuracy**: ML model performance across time horizons
- **System Robustness**: Performance under noise and communication delays

### Visualization Outputs

Publication-quality figures are generated for:

- Charging strategy performance comparisons
- Sensor noise impact analysis
- LSTM prediction accuracy over time horizons
- Communication delay robustness assessment
- Carbon emission optimization results
- Sensitivity analysis across key parameters

## Technical Implementation

### Architecture Components

- **Discrete-Event Simulation**: SimPy-based manufacturing environment modeling
- **Machine Learning Pipeline**: Ensemble methods with LSTM, Random Forest, and Gaussian Process
- **Optimization Engine**: Multi-objective optimization with hierarchical decision-making
- **Data Processing**: Real-time data integration and preprocessing
- **Visualization Framework**: Automated generation of publication-quality figures

## Experimental Validation

### Reproducibility

All experimental results reported in the paper can be reproduced using:

```bash
# Complete validation suite (matches paper results)
python run_experimental_validation.py

# Individual experiment validation
python run_all_experiments.py
```

### Statistical Analysis

The framework implements rigorous statistical validation:

- 30 replications per experimental scenario
- Statistical significance testing (p < 0.05)
- Confidence interval reporting
- Robust performance metrics across multiple random seeds

### Hardware Requirements

Recommended system specifications for full experimental reproduction:

- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB+ for large-scale experiments
- **Storage**: 5GB+ for result storage
- **Runtime**: 2-4 hours for complete validation suite

## Code Structure

```
simulation/
├── config.py                    # System configuration parameters
├── simulation_model.py          # Core discrete-event simulation
├── optimization_controller.py   # Multi-objective optimization engine
├── experimental_framework.py    # Experimental design and execution
└── experiments/                 # Individual experiment modules
    ├── sensor_noise_experiments.py
    ├── lstm_prediction_experiments.py
    ├── communication_experiments.py
    ├── carbon_emission_experiments.py
    └── sensitivity_analysis_experiments.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ev_charging_digital_twin_2024,
  title={Digital Twin Framework for Intelligent EV Charging Optimization in Smart Manufacturing},
  author={[Authors]},
  journal={[Journal]},
  year={2024},
  note={Under Review}
}
```

## License

This research code is provided under the MIT License for academic and research purposes.
