import numpy as np
import pandas as pd

from src.Market import Market


FREQ_COEFF = {'daily': 1, 'weekly': 5, 'monthly': 22}

def make_scenario(nobs:int, freq:str, market:Market, burn:int, init_value:float|None=None, init_vol:float|None=None, 
                  return_last:bool=False, linear:bool=True):

    r, r_m, m_vol = market.simulate(nobs*FREQ_COEFF[freq], burn, init_value, init_vol)
    
    r = pd.DataFrame(r)

    if freq not in ('daily', 'weekly', 'monthly'):
        raise ValueError("Frequency must be one of 'daily', 'weekly', or 'monthly'")
    
    ii = pd.date_range(start='2024-01-01', periods=len(r.index), freq='B')
    r.index, r_m.index, m_vol.index = ii, ii, ii

    if linear:
        r = np.exp(r) - 1
        r_m = np.exp(r_m) - 1

    if freq == 'monthly':
        rule = 'ME'
    elif freq == 'weekly':
        rule = 'W-FRI'

    if rule is not None: # TODO: Doesnt handle freq='daily'
        scenario = r.resample(rule).sum()
            
        if return_last:
            # NOTE: We only need market when also returning last
            market_return = r_m.resample(rule).sum()

            last_r_m = r_m.resample(rule).last()
            last_m_vol = m_vol.resample(rule).last()

            return scenario.iloc[:nobs], market_return.iloc[:nobs],\
                    last_r_m.iloc[:nobs], last_m_vol[:nobs]
    
    return scenario.iloc[:nobs]

def generate_scenarios(K:int, T:int, N:int, freq:str, market:Market, burn:int, init_value:float|None=None, init_vol:float|None=None):
    scenarios = np.zeros((K,T,N))

    for k in range(K):
        scenarios[k] = make_scenario(T, freq, market, burn, init_value, init_vol)

    return scenarios