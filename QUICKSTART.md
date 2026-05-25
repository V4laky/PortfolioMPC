# Quick Start Guide

Get up and running in 5 minutes.

## 1. Initial Setup (One time)

```bash
# Navigate to project
cd path/to/project

# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

See [SETUP.md](SETUP.md) for detailed platform-specific instructions.

---

## 2. Common Workflows

### Workflow A: Run MPC Simulation with Default Config

```bash
python scripts/mpc_simulation.py --config configs/test_config.yaml
```

**Output**: Results saved to `results/test/`
- `main_test_var99_sims.joblib` - Portfolio trajectories
- `main_test_var99_trades.joblib` - Trading decisions

### Workflow B: Generate New Simulation Dataset

```bash
python scripts/generate_dataset.py --config configs/test_config.yaml
```

**Output**: Generated scenarios saved to `datasets/test/`
- `sim_0000.joblib`, `sim_0001.joblib`, etc.
- `market.joblib` - Market reference data

### Workflow C: Explore Results in Jupyter

```bash
jupyter notebook
```

Then open `notebooks/results.ipynb` to visualize:
- Portfolio performance
- Allocation over time
- Risk metrics

### Workflow D: Explore Analysis Results

```bash
jupyter notebook
```

Open either:
- `notebooks/results.ipynb` - Main analysis and visualization
- `notebooks/Trajectory_analysis.ipynb` - Detailed trajectory analysis

### Workflow E: Learn How to Create Market Instance

```bash
jupyter notebook
```

Open `notebooks/MarketSim_tests.ipynb` to see:
- Example of instantiating a Market class
- Understanding Market_v2 parameters
- Note: Requires a pre-trained GARCH model

### Workflow F: Modify Simulation Parameters

1. Edit `configs/test_config.yaml`:
   ```yaml
   risk : 0.5           # Change portfolio risk target
   return_VAR : 0.99    # Change VaR constraint
   trade_cost : 0.001   # Change transaction costs
   ```

2. Run simulation:
   ```bash
   python scripts/mpc_simulation.py --config configs/test_config.yaml
   ```

3. Compare results with previous runs

---

## 3. Configuration Parameters Explained

### Dataset Generation (`generate_dataset.py`)

```yaml
market_path : results/market_simulation_normal_garch_norm.joblib
# Pre-trained market model file (must exist)

dataset_path : datasets/test
# Where to save generated scenarios

n_steps : 20
# Time periods per simulation

n_simulations : 50
# Number of Monte Carlo simulations to generate

freq : weekly  # Options: daily, weekly, monthly
# Frequency of simulated returns
```

### MPC Simulation (`mpc_simulation.py`)

```yaml
# Horizon Parameters
T : 4           # Planning horizon (time periods ahead)
K : 100         # Scenarios per period (more = better but slower)

# Risk Parameters
risk_type : "variance"  # Options: "variance", "cvar"
risk : 0.5              # Risk target (portfolio variance)
cvar_alpha : 0.90       # Confidence for CVaR (ignored if variance)

# Constraints
return_VAR : 0.99       # Min wealth retention (99% = max 1% loss)
                        # Set to None to disable

# Costs
trade_cost : 0.001      # Transaction cost (0.1% = 10bps)

# Model Robustness
misspecification_alpha : 0.0
# Noise for model misspecification testing (0 = perfect model)
```

---

## 4. Typical Workflows by Use Case

### Use Case: Compare Risk Strategies

1. Create copies of config:
   ```bash
   cp configs/test_config.yaml configs/low_risk.yaml
   cp configs/test_config.yaml configs/high_risk.yaml
   ```

2. Edit parameters:
   - `low_risk.yaml`: set `risk: 0.2`
   - `high_risk.yaml`: set `risk: 0.8`

3. Run both:
   ```bash
   python scripts/mpc_simulation.py --config configs/low_risk.yaml
   python scripts/mpc_simulation.py --config configs/high_risk.yaml
   ```

4. Compare results in `results/` folder

### Use Case: Test Model Misspecification

1. Create config:
   ```bash
   cp configs/test_config.yaml configs/misspec_test.yaml
   ```

2. Add misspecification noise:
   ```yaml
   misspecification_alpha : 0.05  # 5% noise on parameters
   name : 'misspec_test_05'
   ```

3. Run:
   ```bash
   python scripts/mpc_simulation.py --config configs/misspec_test.yaml
   ```

### Use Case: Optimize Performance

1. Increase scenarios for better optimization:
   ```yaml
   K : 500  # More scenarios (slower but better)
   ```

2. Extend planning horizon:
   ```yaml
   T : 8    # Longer horizon
   ```

3. Adjust risk to find sweet spot:
   ```yaml
   risk : 0.5  # Try different values
   ```

---

## 5. Understanding Output Files

After running `mpc_simulation.py`, check `results/test/`:

```
main_test_var99_sims.joblib
├── Portfolio wealth over time
├── Shape: (n_simulations, n_timesteps, n_assets+1)
└── Last dim: [cash, asset1, asset2, ...]

main_test_var99_trades.joblib
├── Trading decisions (before transaction costs)
├── Shape: (n_simulations, n_timesteps, n_assets+1)
└── Positive = buy, negative = sell
```

Load and inspect:
```python
from joblib import load

sims = load('results/test/main_test_var99_sims.joblib')
trades = load('results/test/main_test_var99_trades.joblib')

print(f"Simulations shape: {sims.shape}")
print(f"Final wealth stats: mean={sims[:,-1,0].mean()}, std={sims[:,-1,0].std()}")
```

---

## 6. Troubleshooting Quick Fixes

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Activate virtual env: `.venv\Scripts\Activate.ps1` |
| Script doesn't run | Run from project root: `cd path/to/project` |
| Old results mixed up | Check `name` in config, create new output name |
| Slow performance | Reduce `K` (scenarios), reduce `n_simulations` |
| Memory issues | Reduce dataset size: fewer `n_simulations` in generate_dataset |

---

## 7. Next Steps

1. ✅ Complete setup
2. 🎯 Run Workflow A (default MPC)
3. 📊 Open `notebooks/results.ipynb` to see results
4. 🔧 Modify parameters in config
5. 📈 Explore other workflows

**For detailed information**, see:
- [README.md](README.md) - Full project documentation
- [SETUP.md](SETUP.md) - Detailed setup for your OS
- `notebooks/` - Interactive examples and analysis
