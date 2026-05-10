import numpy as np
import pandas as pd

from src.Market import log_norm_params, get_uncond_vol, transform_params


class Market():
    
    def __init__(self, n_assets, n_sectors, market_model, rf = 0.0001, market_scale_arch=1,
                 random_seed=None, target_SNR=None, base_w = np.array([0.4, 0.3, 0.2, 0.1])):
        
        assert np.isclose(base_w.sum(), 1), "base_w must sum to 1."
        assert (base_w >= 0).all(), "base_w must be non-negative."

        if target_SNR is not None:
            raise NotImplementedError("targer SNR is not implemented yet.")

        np.random.seed(random_seed)

        self.market_model = market_model
        self.scale = market_scale_arch
        self.uncond_vol = get_uncond_vol(self.market_model, self.scale, 1)
        self.params = transform_params(market_model.params, 0.85, self.uncond_vol, self.scale)

        # estimate paramters
        r_m = self.market_model.model.simulate(nobs=20000, params=self.params)
        self.mean_vol = r_m['volatility'].mean() / self.scale
        self.vol_of_vol = (r_m['volatility'] / self.scale).std()

        smooth_vol = self._calc_smooth_factor(r_m['volatility'] / self.scale, 'ewm', 20)

        self.excess_vol_std = ((r_m['volatility'].iloc[20:] / self.scale) - smooth_vol).std()


        self.n_assets = n_assets
        self.n_sectors = n_sectors
        self.rf = rf

        # Defince weights of varinance budget
        self.base_w = base_w

        weights = self.base_w[None, :] + np.random.normal(0, 0.03, (n_assets, 4))
        weights = np.clip(weights, 0.01, None)
        weights /= weights.sum(axis=1, keepdims=True)

        self.weights = weights
        
        # Define the volatilities of individual assets
        self.vols = np.random.lognormal(*log_norm_params(0.00945, 0.00315), n_assets) # residual vol

        # Defince Sectors volatilities and AR coefficients (make it persistent)
        self.sector_AR_coeffs = np.random.uniform(0.2, 0.6, n_sectors)
        self.sectors = np.random.randint(0, n_sectors, n_assets)

        # trying to prevent huge coeffs by lowering std
        self.sector_vols = np.random.lognormal(*log_norm_params(0.00945, 0.002), n_sectors)
        
        # Calculate factor coefficients
        self.betas = self.vols / self.mean_vol * np.sqrt(self.weights[:, 0]) # market
        self.vol_premia = self.vols / self.excess_vol_std * np.sqrt(self.weights[:, 1]) # vol premia
        self.sector_coeffs = self.vols / self.sector_vols[self.sectors] * np.sqrt(self.weights[:, 2])

        self.vol_premia *= np.random.choice([-1, 1], n_assets) # Some assets react negatively to volatility

        # idio volatility
        self.idio_vol = self.vols * np.sqrt(self.weights[:, 3])
        
        
        # alphas
        self.alphas = np.random.normal(0, 0.0001, n_assets) + (self.vol_premia < 0) * 0.0001 # higher alpha for negative vol premia
    

    def summary(self):
        df = pd.DataFrame({'alpha': self.alphas, 'beta': self.betas, 'idio vols': self.vols,
                           'sector': self.sectors, 'sector_coeffs': self.sector_coeffs,
                           'sector_vols': self.sector_vols[self.sectors], 'vol_premium':self.vol_premia})
        return df
    

    def simulate(self, nobs, burn=500, initial_market_returns=None, initial_market_vols=None, initial_sector_returns=None, 
                 return_sectors=False):

        # placeholder
        roll = [20, 20, 20] # market - vol - sector
        roll_type = ['ewm', "ewm", 'ewm']
        extra_gen = max(roll)-1 # generate extra for rolling

        # check shapes
        if initial_market_returns is not None:
            assert initial_market_returns.size == roll[0], f"initial_market_returns.size must be {roll[0]}"
            extra_gen = 0 # Dont need to generate extra 
        if initial_market_vols is not None:
            assert initial_market_vols.size == roll[1], f"initial_market_vols.size must be {roll[1]}"
        if initial_sector_returns is not None:
            assert initial_sector_returns.shape == (roll[2], self.n_sectors), f"initial_sector_returns.shape must be {(roll[2], self.n_sectors)}"

        # Market returns, and vols
        initial_value = initial_market_returns.iloc[-1] * self.scale if initial_market_returns is not None else None
        initial_vol = initial_market_vols.iloc[-1] * self.scale if initial_market_vols is not None else None

        r_m = self.market_model.model.simulate(nobs=nobs+1+extra_gen, params=self.params, burn=burn,
                                                    initial_value=initial_value, initial_value_vol=initial_vol)
                
        r_m = r_m.iloc[1:] / self.scale
        r_m, m_vol = r_m['data'], r_m['volatility']

        #m_vol -= self.mean_vol # center volatility so its not > 0
        # NOTE:tryinf m_vol - rolling mean(m_vol)

        # Sectors

        if initial_sector_returns is None:
            sector_returns = self._simulate_sector(nobs+(roll[2]-1), initial_sector_returns).T
        else:
            sector_returns = self._simulate_sector(nobs, initial_sector_returns).T

        # concat with initials if not None
        if initial_market_returns is not None:
            r_m = pd.concat([initial_market_returns, r_m])
        if initial_market_vols is not None:
            m_vol = pd.concat([initial_market_vols, m_vol])
        if initial_sector_returns is not None:
            sector_returns = np.hstack([initial_sector_returns.T, sector_returns])
        

        # smooth factors
        market_factor = self._calc_smooth_factor(pd.Series(r_m), roll_type[0], roll[0])
        market_vol_factor = self._calc_smooth_factor(pd.Series(m_vol), roll_type[1], roll[1])
        sector_factor = self._calc_smooth_factor(pd.DataFrame(sector_returns.T), roll_type[2], roll[2])

        # Use the fact that the last one is always time T, so use last nobs
        market_factor = market_factor.iloc[-nobs:]
        market_vol_factor = m_vol[-nobs:] - market_vol_factor.iloc[-nobs:] # excess vol from recent
        sector_factor = sector_factor.iloc[-nobs:]

        # construct returns
        returns = self.alphas[:, None] + self.betas[:, None] @ (market_factor.to_numpy()[None, :] - self.rf) + self.rf
        returns += self.vol_premia[:, None] @ market_vol_factor.to_numpy()[None, :]
        returns += (sector_factor.T).iloc[self.sectors] * self.sector_coeffs[:, None]
        returns += np.random.standard_normal((self.n_assets, nobs)) * self.idio_vol[:, None]

        if return_sectors:
            return returns.T, r_m.iloc[-nobs:], m_vol.iloc[-nobs:], sector_returns[:, -nobs:]
        return returns.T, r_m.iloc[-nobs:], m_vol.iloc[-nobs:]

    def _simulate_sector(self, nobs, initial_sector_returns=None):
        
        shock = np.random.standard_normal((nobs, self.n_sectors)) * self.sector_vols * np.sqrt(1-self.sector_AR_coeffs**2) 

        sector_returns = np.zeros((nobs, self.n_sectors))

        prev_r = initial_sector_returns[-1, :] if initial_sector_returns is not None else np.zeros(self.n_sectors)

        for i in range(nobs):
            sector_returns[i] = self.sector_AR_coeffs * prev_r + shock[i]
            prev_r = sector_returns[i]
        
        return sector_returns

    def _calc_smooth_factor(self, raw_factor, roll_type, roll_lenght):
        if roll_type == "rolling mean":
            smooth_factor = raw_factor.rolling(roll_lenght).mean().dropna()

        elif roll_type == 'ewm':
            smooth_factor = raw_factor.ewm(span=roll_lenght, adjust=False, min_periods=roll_lenght).mean().dropna()

        elif roll_type is not None:
            raise ValueError("Roll type must be one of the following: rolling mean, ewm, None")
        
        return smooth_factor
