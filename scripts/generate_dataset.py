import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import yaml

from src.simulation_generation import generate_dataset
from joblib import load, dump

def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file"
    )

    args = parser.parse_args()
    
    
    # Get config values

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    
    market = load(project_root / config['market_path'])
    dataset_path = project_root / config['dataset_path']

    N = market.n_assets # n_assets
    T = config['T'] # n_timeperiods - the amount the model sees ahead
    K = config['K'] # n_scenarios

    freq = config['freq']
    n_steps = config['n_steps']
    n_simulations = config['n_simulations'] # Maybe a startfrom could be nice later?

    dataset_path.mkdir(parents=True, exist_ok=True)

    generate_dataset(n_simulations, n_steps, freq, market, K,T,N, dataset_path)

if __name__ == '__main__':
    main()