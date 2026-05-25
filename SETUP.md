# Setup Guide - Getting Started

This guide walks through setting up the project on Windows, macOS, and Linux.

## Quick Start (All Platforms)

```bash
# 1. Navigate to project directory
cd path/to/project

# 2. Create virtual environment
python -m venv .venv

# 3. Activate it (see platform-specific steps below)

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run a test
python scripts/mpc_simulation.py --config configs/test_config.yaml
```

---

## Platform-Specific Instructions

### Windows (PowerShell)

```powershell
# 1. Open PowerShell and navigate to project
cd C:\path\to\your\project

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
.venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# Then try activation again:
.venv\Scripts\Activate.ps1

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Verify installation
python -c "import pandas; import numpy; import cvxpy; print('Setup successful!')"
```

**Troubleshooting**:
- If PowerShell execution policy error persists, use Command Prompt (cmd.exe) instead and run `.venv\Scripts\activate.bat`
- If pip install fails, try: `python -m pip install --upgrade pip setuptools wheel`

### Windows (Command Prompt)

```cmd
# 1. Navigate to project
cd C:\path\to\your\project

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
.venv\Scripts\activate.bat

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Verify
python -c "import pandas; import numpy; import cvxpy; print('Setup successful!')"
```

### macOS

```bash
# 1. Navigate to project
cd /path/to/your/project

# 2. Check Python version (requires 3.8+)
python3 --version

# 3. Create virtual environment
python3 -m venv .venv

# 4. Activate virtual environment
source .venv/bin/activate

# 5. Upgrade pip
python -m pip install --upgrade pip

# 6. Install dependencies
pip install -r requirements.txt

# 7. Verify installation
python -c "import pandas; import numpy; import cvxpy; print('Setup successful!')"
```

**Note**: If you have both Python 2 and 3 installed, always use `python3` and `pip3` explicitly.

### Linux (Ubuntu/Debian)

```bash
# 1. Update package manager
sudo apt-get update

# 2. Ensure Python 3 and pip are installed
sudo apt-get install python3 python3-pip python3-venv

# 3. Navigate to project
cd /path/to/your/project

# 4. Create virtual environment
python3 -m venv .venv

# 5. Activate virtual environment
source .venv/bin/activate

# 6. Upgrade pip
python -m pip install --upgrade pip

# 7. Install dependencies
pip install -r requirements.txt

# 8. Verify installation
python -c "import pandas; import numpy; import cvxpy; print('Setup successful!')"
```

### Linux (Fedora/RHEL/CentOS)

```bash
# 1. Update package manager
sudo dnf update

# 2. Install dependencies
sudo dnf install python3 python3-pip python3-devel gcc

# 3. Navigate to project
cd /path/to/your/project

# 4. Create virtual environment
python3 -m venv .venv

# 5. Activate virtual environment
source .venv/bin/activate

# 6. Upgrade pip
python -m pip install --upgrade pip

# 7. Install dependencies
pip install -r requirements.txt

# 8. Verify installation
python -c "import pandas; import numpy; import cvxpy; print('Setup successful!')"
```

---

## Common Issues and Solutions

### Issue: `python` command not found

**Solution**: Use `python3` instead
```bash
python3 -m venv .venv
```

### Issue: `pip: command not found`

**Solution**: Use `python -m pip` instead
```bash
python -m pip install -r requirements.txt
```

### Issue: Permission denied on activation script

**Linux/macOS**: Make sure the script is executable:
```bash
chmod +x .venv/bin/activate
source .venv/bin/activate
```

### Issue: Module not found (e.g., `ModuleNotFoundError: No module named 'cvxpy'`)

**Solution**:
1. Verify virtual environment is activated (should see `(.venv)` in terminal)
2. Reinstall requirements:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Issue: CVXPY solver issues

**Solution**: Some systems need additional solver libraries:
```bash
# Linux
sudo apt-get install libopenblas-dev liblapack-dev libgomp1

# macOS
brew install openblas lapack

# Then reinstall
pip install --upgrade --force-reinstall cvxpy cvxopt
```

### Issue: Matplotlib backend issues in Jupyter

**Solution**: Run this in first notebook cell:
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

---

## Verifying Your Setup

After installation, run these tests:

```python
# Test 1: Basic imports
python -c "
import pandas, numpy, scipy, statsmodels, numba
import matplotlib, seaborn, plotly
import arch, cvxpy, cvxopt
import scikit-learn, yaml
print('✓ All basic imports successful')
"

# Test 2: Run configuration
python scripts/generate_dataset.py --config configs/test_config.yaml

# Test 3: Run MPC simulation
python scripts/mpc_simulation.py --config configs/test_config.yaml
```

If all three complete without errors, your setup is ready!

---

## Next Steps

1. **Review the README.md** for project overview and usage
2. **Run an MPC simulation** with `scripts/mpc_simulation.py --config configs/test_config.yaml`
3. **Explore results**: Open `notebooks/results.ipynb` to visualize output
4. **Analyze trajectories**: See `notebooks/Trajectory_analysis.ipynb` for detailed analysis
5. **Modify configs/test_config.yaml** to customize simulations
6. **Generate new datasets** with `scripts/generate_dataset.py` if needed

---

## Deactivating Virtual Environment

When you're done working:

```bash
# All platforms
deactivate
```

To reactivate later:
- **Windows PowerShell**: `.venv\Scripts\Activate.ps1`
- **Windows CMD**: `.venv\Scripts\activate.bat`
- **macOS/Linux**: `source .venv/bin/activate`

---

## Using Conda Instead of venv

If you prefer conda:

```bash
# Create environment
conda create -n szakdoga python=3.10

# Activate
conda activate szakdoga

# Install
pip install -r requirements.txt
```
