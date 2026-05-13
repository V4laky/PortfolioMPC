import numpy as np
import pandas as pd

from src.Market import Market


FREQ_COEFF = {'daily': 1, 'weekly': 5, 'monthly': 22}

def make_scenario(nobs:int, freq:str, market:Market, return_full_daily:bool=False,
                    linear:bool=True, **market_kwargs):
    """NOTE: linear applies only to scenarios."""
    
    sectors = None

    # Determine whether to return sectors
    if 'return_sectors' in market_kwargs and market_kwargs['return_sectors']:
            r, r_m, m_vol, sectors = market.simulate(nobs*FREQ_COEFF[freq], **market_kwargs)
    else:
        r, r_m, m_vol = market.simulate(nobs*FREQ_COEFF[freq], **market_kwargs)
        
    # Handle empty r_m, m_vol
    if r_m is None:
        r_m = pd.Series(np.zeros(r.shape[0]))
    if m_vol is None:
        m_vol = pd.Series(np.zeros(r.shape[0]))
    
    r = pd.DataFrame(r)

    if freq not in ('daily', 'weekly', 'monthly'):
        raise ValueError("Frequency must be one of 'daily', 'weekly', or 'monthly'")
    
    ii = pd.date_range(start='2024-01-01', periods=len(r.index), freq='B')
    r.index, r_m.index, m_vol.index = ii, ii, ii

    if freq == 'monthly':
        rule = 'ME'
    elif freq == 'weekly':
        rule = 'W-FRI'

    # Make scenarios from log-returns.
    if freq != 'daily':
        scenario = r.resample(rule).sum()
    else:
        scenario = r
    
    # Make scenarios linear AFTER making them
    if linear:
        scenario = np.exp(scenario) - 1
    
    # Handle more detailed return
    if return_full_daily:
        if freq != 'daily':
            market_return = r_m.resample(rule).sum()
        else:
            market_return = r_m

        return scenario, market_return, {'market': r_m, 'market_vol':m_vol, 'sectors': sectors}

    return scenario.iloc[:nobs]

def generate_scenarios(K:int, T:int, N:int, freq:str, market:Market, **market_kwargs):
    scenarios = np.zeros((K,T,N))

    for k in range(K):
        scenarios[k] = make_scenario(T, freq, market, **market_kwargs)

    return scenarios


from scipy.optimize import root_scalar

# Docsting written by chatgpt
def simulate_target_weight_portfolio(x0, w_target, rf, returns, trading_cost):
    """
    Simulate a target-weight portfolio under proportional transaction costs,
    using a "trade-then-hold" timing convention and a wealth-shrinkage cost model.

    Key Modeling Assumptions
    -----------------------
    1. Timing Convention (Nonstandard):
       At each period t:
         a) The portfolio is rebalanced FIRST to the target weights.
         b) Then asset returns for period t are realized.
       This differs from the more common "returns-then-rebalance" convention.

       Interpretation:
         - The portfolio enters each period already at target weights.
         - This is especially convenient when x0 is fully in cash, avoiding
           an initial idle period before investment.

    2. Transaction Costs:
       - Costs are proportional to L1 turnover in risky assets only:
             cost = c * ||Δx_risky||_1
       - Cash (asset 0) is assumed frictionless (no cost to hold or transfer).
       - Costs do NOT come from a cash account; instead, they reduce total wealth.

    3. Wealth Shrinkage Model:
       - After rebalancing, the portfolio exactly matches target weights:
             x_post = l * w_target
       - The scalar l < total pre-trade wealth reflects transaction costs:
             l = V_pre - cost
       - This allows exact rebalancing at every step without feasibility issues.

    4. Assets:
       - Asset 0 is cash with return `rf`
       - Remaining assets are risky, with returns given by `returns`

    Parameters
    ----------
    x0 : array-like (N,)
        Initial portfolio in dollar terms. Typically all cash.
    w_target : array-like (N,)
        Target portfolio weights (must sum to 1).
        Includes cash as the first component.
    rf : float or array-like
        Risk-free rate per period for the cash asset.
    returns : pandas.DataFrame (T x (N-1))
        Time series of returns for risky assets.
    trading_cost : float
        Proportional transaction cost coefficient (per unit L1 turnover).

    Returns
    -------
    sims : ndarray (T+1, N)
        Simulated portfolio positions over time.
        sims[0] is the initial portfolio (pre-first rebalance).

    Notes
    -----
    - The portfolio is always exactly at target weights immediately before returns.
    - Transaction costs appear as reductions in total portfolio value.
    - This model is well-suited for comparing strategies under identical
      turnover-based cost assumptions, since it avoids path-dependent
      feasibility constraints (e.g., cash shortages).
    
    """
    
    x_prev = x0.copy()
    sims = [x_prev]

    if isinstance(w_target, list):
        w_target = np.array(w_target)

    for i, col in enumerate(returns.columns):
        x_prev = sims[-1]

        # We shrink total wealth from W_pre to `l` to pay for trading,
        # and choose `l` so that the cost of moving from x_prev to l*w_target
        # exactly matches that shrinkage.

        def f(l):
            return -l + sum(x_prev) - trading_cost * np.linalg.norm((l * w_target - x_prev)[1:], 1)

        l_res = root_scalar(f, method='newton', x0=sum(x_prev)).root

        x_target = l_res * w_target

        sims.append(x_target * np.hstack([1+rf, 1+returns[col].values]))

    return np.array(sims)
