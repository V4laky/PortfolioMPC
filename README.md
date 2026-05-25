# Market Simulation & Model Predictive Control

A Python-based project for financial market simulation and model predictive control (MPC) portfolio optimization.

## Project Overview

This project implements:
- **Market Simulation**: Realistic financial market simulation using GARCH models and factor models
- **Portfolio Optimization**: Model predictive control for portfolio allocation with risk constraints
- **Scenario Generation**: Multi-scenario generation for robust portfolio planning

## Project Structure

```
├── src/
│   ├── Market.py              # Deprecated - use Market_v2.py instead
│   ├── Market_v2.py           # Primary market model (factor-based with GARCH)
│   ├── MPCController.py       # Model predictive control for portfolio management
│   ├── portfolio_analysis.py  # Portfolio analysis utilities
│   ├── simulation_generation.py# Dataset generation from market simulations
│   ├── optimized_functions.py # Performance-optimized functions (GARCH, EWMA)
│   └── utils.py               # Utility functions for scenarios and data processing
├── scripts/
│   ├── generate_dataset.py    # Generate simulation datasets
│   └── mpc_simulation.py      # Run MPC simulations on datasets
├── notebooks/
│   ├── results.ipynb          # Main results analysis and visualization
│   ├── Trajectory_analysis.ipynb # Portfolio trajectory and performance analysis
│   ├── MarketSim_tests.ipynb  # Example: Creating Market class instances
│   ├── factors_v2.ipynb       # (Deprecated - no longer maintained)
│   └── convex_opt.ipynb       # (Deprecated - old version, use scripts instead)
├── configs/
│   └── test_config.yaml       # Configuration for MPC simulation
├── datasets/
│   └── test/                  # Test datasets (simulated market scenarios)
├── results/                   # Output directory for simulations and plots
└── requirements.txt           # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8 or later
- pip or conda

### Setup

1. **Clone the repository** (or download/extract the project folder)

2. **Create and activate a virtual environment**:

   **Windows (PowerShell)**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   **Windows (Command Prompt)**:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate.bat
   ```

   **Linux/macOS**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation** by running a quick test:
   ```bash
   python scripts/mpc_simulation.py --config configs/test_config.yaml
   ```

## Usage

### Generating Simulation Datasets

Create a new dataset of market simulations:

```bash
python scripts/generate_dataset.py --config configs/test_config.yaml
```

**Configuration** (`configs/test_config.yaml`):
- `market_path`: Path to pre-trained market model (joblib format)
- `dataset_path`: Output directory for generated scenarios
- `n_steps`: Number of time steps per simulation
- `n_simulations`: Number of Monte Carlo simulations
- `freq`: Sampling frequency ('daily', 'weekly', or 'monthly')

### Running MPC Simulations

Execute model predictive control on generated datasets:

```bash
python scripts/mpc_simulation.py --config configs/test_config.yaml
```

**Configuration options** in `configs/test_config.yaml`:
- `K`: Number of scenarios for lookahead
- `T`: Planning horizon (time periods)
- `N`: Number of assets (inferred from market model)
- `risk`: Risk parameter (portfolio variance target)
- `risk_type`: 'variance' or 'cvar'
- `cvar_alpha`: Confidence level for CVaR constraint
- `return_VAR`: Value-at-Risk constraint (minimum wealth retention)
- `trade_cost`: Transaction cost (e.g., 0.001 for 10bps)
- `misspecification_alpha`: Model misspecification noise level

### Jupyter Notebooks

Explore the project through interactive notebooks:

```bash
jupyter notebook
```

Key notebooks:
- `notebooks/results.ipynb` - Main analysis: visualize MPC optimization results
- `notebooks/Trajectory_analysis.ipynb` - Analyze portfolio trajectories and performance
- `notebooks/MarketSim_tests.ipynb` - Example of how to create a Market class instance (requires GARCH model)

## Dependencies

Core dependencies (see `requirements.txt`):
- **Data Processing**: pandas, numpy, scipy
- **Statistics**: statsmodels, scikit-learn, arch
- **Optimization**: cvxpy, cvxopt
- **Performance**: numba, tqdm
- **Visualization**: matplotlib, seaborn, plotly
- **Notebooks**: ipykernel, ipywidgets
- **Finance**: yfinance
- **Config**: pyyaml

## Key Classes and Functions

### Market Models
- `Market` (Market_v2.py): Primary factor-based market model with GARCH dynamics
- `Market` (Market.py): Deprecated - use Market_v2.py instead

### Optimization
- `MPCController`: Model predictive control for portfolio allocation

### Utilities
- `make_scenario()`: Generate market scenarios at specified frequency
- `generate_dataset()`: Create multi-scenario datasets
- `simulate_gjr_garch()`: GJR-GARCH market returns simulation

## Data Files

**Required pre-trained models** (included in results/):
- `market_simulation_normal_garch_norm.joblib`: Pre-trained market model
- `market_model_normal_garch_norm.joblib`: Market model parameters

**Generated data** (created during runs):
- `datasets/test/`: Simulation scenarios
- `results/test/`: MPC optimization results and trading decisions

## Troubleshooting

### Import Errors
If you get import errors when running scripts, ensure:
1. Virtual environment is activated
2. All dependencies are installed: `pip install -r requirements.txt`
3. You're running scripts from the project root directory

### CVXPY Solver Issues
If optimization fails, try:
1. Ensure cvxopt is properly installed: `pip install --upgrade cvxopt`
2. Check configuration parameters are valid (risk > 0, T > 0, etc.)

### Path Handling (IMPORTANT)
The project uses relative paths from the project root. Always:
1. Run scripts from the project root directory
2. Use absolute paths or relative paths from project root in config files

**Always** use relative paths from project root:

✅ **Correct**:
```python
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "datasets" / "test"
```

❌ **Incorrect**:
```python
# Don't hardcode your PC's paths
data_path = "C:\\Users\\YourName\\Desktop\\Szakdoga\\code\\datasets"

# Don't use absolute filesystem paths
data_path = "/home/user/project/datasets"
```

### Configuration Files

YAML config paths must be relative to project root:

```yaml
# ✅ Correct
market_path : results/market_simulation_normal_garch_norm.joblib
dataset_path : datasets/test
save_path : results/test

# ❌ Incorrect
market_path : C:/Users/lovas/Desktop/Szakdoga/code/results/...
```



## ⚠️ Deprecated Files - Do Not Use

The following files are no longer maintained:
- **Market.py** - Deprecated. Use `Market_v2.py` instead
- **convex_opt.ipynb** - Old version. Use `scripts/mpc_simulation.py` instead
- **factors_v2.ipynb** - No longer maintained

## Notes

- **Relative Paths**: All file paths are relative to the project root for portability across machines
- **Large Files**: Market models and datasets are not version-controlled (in .gitignore) - ensure pre-trained models are available before running simulations
- **Numerical Stability**: The project uses numba-compiled functions for GARCH/EWMA calculations for performance

## Author Notes

This is a research/thesis project. The code implements:
- Realistic market dynamics with conditional volatility (GARCH)
- Factor-based asset returns with sector effects
- Robust MPC for portfolio management under model misspecification
- Multiple risk metrics (variance, CVaR, VaR constraints)

---

For questions or issues, please refer to the notebook examples for detailed workflows.
