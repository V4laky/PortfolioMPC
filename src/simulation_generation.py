import numpy as np
import pandas as pd
import math

import os
from joblib import load, dump

from src.utils import FREQ_COEFF, make_scenario
MAGIC_NUMBER=20

def generate_dataset(n_simulations, n_steps, freq, market, K,T,N, outputdir):

        dump(market, outputdir  / "market.joblib")

        for simulation in range(n_simulations):
            
            seed = None # Placeholder for later

            extra = math.ceil(MAGIC_NUMBER / FREQ_COEFF[freq])
            actual_traj, market_return, context = \
                make_scenario(n_steps + extra, freq, market, 
                              return_full_daily=True, return_sectors=True)

            # first were generated for context - leave one extra for previous index
            actual_traj = actual_traj.iloc[-(n_steps+1):] 
            market_return = market_return.iloc[-(n_steps+1):]

            # reindex and transpose sectors
            full_date_index = context['market'].index
            context['sectors'] = pd.DataFrame(context['sectors'].T, index=full_date_index)

            sim_data = {
                'actual_traj': actual_traj,
                'market_return': market_return,
                'context': context,
                'params': {
                    'freq': freq,
                    'MAGIC_NUMBER': MAGIC_NUMBER,
                    'N': N,
                    'K': K,
                    'T': T,
                },
                'seed': seed,
            }

            dump(sim_data, outputdir / f"sim_{str(simulation).zfill(4)}.joblib")


def load_simulation(path, sim_num):
    """
    Returns actual_traj, market_return, context.
    """
    file_path = path / f"sim_{str(sim_num).zfill(4)}.joblib"
    
    sim_data = load(file_path)

    return sim_data['actual_traj'], sim_data['market_return'], sim_data['context']
