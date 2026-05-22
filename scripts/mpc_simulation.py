import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import numpy as np
import pandas as pd

from joblib import dump, load
import yaml

from src.MPCController import MPCController
from src.simulation_generation import generate_dataset, load_simulation
from src.utils import make_scenario, generate_scenarios, FREQ_COEFF

MAGIC_NUMBER = 20

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
    save_path = project_root / config['save_path']

    filename = config['name']

    N = market.n_assets # n_assets
    T = config['T'] # n_timeperiods - the amount the model sees ahead
    K = config['K'] # n_scenarios

    freq = config['freq']
    rf = market.rf

    n_steps = config['n_steps']
    n_simulations = config['n_simulations'] # Maybe a startfrom could be nice later?

    trade_cost = config['trade_cost'] # 5bps - 20bps
    risk = config['risk']

    controller = MPCController(K, T, N, risk_free=rf, risk=risk, trade=trade_cost, 
                               risk_type=config['risk_type'], cvar_alpha=config['cvar_alpha'])

    # MPC simulation

    from tqdm import tqdm
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        x_init = [100] + [0 for _ in range(N)]

        MPC_sims = np.zeros((n_simulations, n_steps+1, N+1)) # + initial value
        MPC_trades = np.zeros((n_simulations, n_steps, N+1))

        for i in tqdm(range(n_simulations), desc='Simulations', position=0):

            actual_traj, market_return, context = load_simulation(dataset_path, i)

            MPC_sim = [x_init]
            MPC_trade = []

            for step in tqdm(range(n_steps), desc=f'simulation {i+1}', leave=False, position=1):

                start = step * FREQ_COEFF[freq]

                scenarios = generate_scenarios(K ,T, N, freq, market, burn=0,
                                            initial_market_returns=context['market'][start:start+MAGIC_NUMBER],
                                            initial_market_vols=context['market_vol'][start:start+MAGIC_NUMBER], 
                                            initial_sector_returns=context['sectors'][start:start+MAGIC_NUMBER, :])
                
                controller.update_parameters(scenarios, MPC_sim[-1])
                stats = controller.solve(solver='OSQP')
                u0 = controller.control()

                x_next = (MPC_sim[-1] + u0) * np.hstack([1+rf, 1+actual_traj[step]])

                MPC_sim.append(x_next)
                MPC_trade.append(u0)

            MPC_sims[i] = np.array(MPC_sim)
            MPC_trades[i] = np.array(MPC_trade)


    save_path.mkdir(parents=True, exist_ok=True)
    dump(MPC_sims, save_path / f"{filename}_sims.joblib")
    dump(MPC_trades, save_path / f"{filename}_trades.joblib")


if __name__ == "__main__":
    main()