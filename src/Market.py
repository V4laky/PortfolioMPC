import numpy as np
import pandas as pd

def log_norm_params(mu, sig):
    """Convert from the lognormal paramters to the ones of the underlying normal."""
    mu_norm = np.log(mu**2 / np.sqrt(mu**2 + sig**2))
    sig_norm = np.sqrt(np.log(1 + (sig**2 / mu**2)))

    return mu_norm, sig_norm

def get_uncond_vol(market_garch, garch_scaling, shrink=0.9):
    p = market_garch.params

    if 'gamma[1]' in p:
        uncond_var = p['omega']/(1-p['alpha[1]']*shrink - p['beta[1]']*shrink - p['gamma[1]']*shrink/2)
    else:
        uncond_var = p['omega']/(1-p['alpha[1]']*shrink - p['beta[1]']*shrink) # shrink parameters since a+b close to 1

    return np.sqrt(uncond_var)/garch_scaling

def transform_params(params, persistance, uncond_vol, garch_scaling):
    p = params.copy()
    pers = p['alpha[1]'] + p['beta[1]'] + p['gamma[1]']/2

    scaling = persistance / pers

    p['alpha[1]'], p['beta[1]'], p['gamma[1]'] = p['alpha[1]']*scaling, p['beta[1]']*scaling, p['gamma[1]']*scaling
    p['omega'] = (uncond_vol * garch_scaling)**2 * (1 - persistance) # uncodnd vol remains same

    return p


# Market return around 0.00028

class Market():
    
    def __init__(self, n_assets, n_sectors, market_model, rf = 0.0001, market_scale_arch=1, 
                 dynamic_beta=True):

        np.random.seed(42)

        self.market_model = market_model

        self.scale = market_scale_arch

        self.dynamic_beta = dynamic_beta
        self.uncond_market_vol = get_uncond_vol(self.market_model, self.scale, 1)

        self.params = transform_params(self.market_model.params, 0.95, self.uncond_market_vol, self.scale)

        self.n_assets = n_assets
        self.n_sectors = n_sectors
        self.rf = rf

        self.alphas = np.random.normal(0, 0.0001, n_assets)
        self.betas = np.random.normal(1, 0.2, n_assets)

        self.vols = np.random.lognormal(*log_norm_params(0.00945, 0.00315), n_assets) # residaul vol

        self.sectors = np.random.randint(0, n_sectors, n_assets)

        self.sector_vols = np.random.lognormal(*log_norm_params(0.006, 0.002), n_sectors)
        
        self.u = np.random.uniform(0.35, 0.45, n_assets) # The amount of variance explained by sectors in %

        self.sector_coeffs = np.sqrt(self.u) * (self.vols / self.sector_vols[self.sectors])
        self.vols *= np.sqrt(1 - self.u) # remaining vol
    
    def summary(self):
        df = pd.DataFrame({'alpha': self.alphas, 'beta': self.betas, 'vols': self.vols,
                           'sector': self.sectors, 'sector_coeffs': self.sector_coeffs,
                           'sector_vols': self.sector_vols[self.sectors]})
        return df

    def simulate(self, nobs, burn=500, initial_value=None, initial_vol=None):

        # NOTE: test scaling up initials to match garch, Possibly a really bad bug !!
        initial_value = initial_value * self.scale if initial_value is not None else None
        initial_vol = initial_vol * self.scale if initial_vol is not None else None

        r_m = self.market_model.model.simulate(nobs=nobs+1, params=self.params, burn=burn,
                                               initial_value=initial_value, initial_value_vol=initial_vol)
        
        r_m = r_m.iloc[1:] / self.scale
        r_m, m_vol = r_m['data'], r_m['volatility']

        sector_returns = np.zeros((self.n_sectors, nobs))

        for i in range(self.n_sectors):
            sector_returns[i] = np.random.standard_t(5, nobs) * self.sector_vols[i] * np.sqrt((5-2)/5)


        if self.dynamic_beta:
            scale = 1 + 0.1*(m_vol / self.uncond_market_vol - 1)
            scale = np.clip(scale, 1.0, 2.0) # lower 1.0 to not reduce betas
            
            betas = self.betas[:, None] @ (scale).to_numpy()[None, :]
            
            returns = self.alphas[:, None] + betas * (r_m.to_numpy()[None, :] - self.rf) + self.rf
        else:
            returns = self.alphas[:, None] + self.betas[:, None] @ (r_m.to_numpy()[None, :] - self.rf) + self.rf

        
        returns += sector_returns[self.sectors] * self.sector_coeffs[:, None]
        returns += np.random.standard_normal((self.n_assets, nobs)) * self.vols[:, None]

        return returns.T, r_m, m_vol
                