import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_simulations(dfs:list[pd.DataFrame], labels):
    fig, ax = plt.subplots(1, len(labels), figsize=(16, 6))
    
    low = min([df.min().min() for df in dfs])
    high = max([df.max().max() for df in dfs])

    low = 0.9*low if low > 0 else 1.1*low
    high = 1.1*high if high > 0 else 0.9*high

    for i, (label, df) in enumerate(zip(labels, dfs)):    
        ax[i].plot(df.mean(), label='mean')
        ax[i].fill_between(df.columns, df.mean() - df.std(), df.mean() + df.std(), alpha=0.2, label="mean ± std")
        ax[i].plot(df.max(), label = 'max', linestyle="--", color='green')
        ax[i].plot(df.min(), label = 'min', linestyle="--", color="red")

        ax[i].set_title(f"{label}")
        ax[i].set_ylim(0.9*low, 1.1*high)

        ax[i].legend()


def compute_returns(dfs:list[pd.DataFrame], names:list[str]):
    returns = {}

    for name, df in zip(names, dfs):
        returns[name] = (df.T / df.T.shift(1)) - 1
        returns[name].dropna(inplace=True)
    return returns

from scipy.stats import ttest_1samp
def conf_int(ser:pd.Series, conf_level=0.95):
    # test if its alpha or beta
    if ser.mean() < 0.05:
        res = ttest_1samp(ser, 0, alternative='greater')
    else:
        res = ttest_1samp(ser, ser.mean())
    
    conf = res.confidence_interval(conf_level)
    low, high = conf.low.round(6), conf.high.round(6)
    
    return (low, high, res.pvalue.round(6))

def conf_int_low(ser:pd.DataFrame, conf_level=.95):
    return conf_int(ser, conf_level)[0]

def conf_int_high(ser:pd.DataFrame, conf_level=.95):
    return conf_int(ser, conf_level)[1]

def conf_int_p(ser:pd.DataFrame, conf_level=.95):
    return conf_int(ser, conf_level)[2]

from scipy.stats import t
def compute_from_simulation_level(ser:pd.Series, conf_level=0.95, name=""):
    sample_mean = ser.mean()
    sample_std = ser.std()
    sample_se = ser.sem()

    t_crit = t.ppf(1 - (1-conf_level)/2, df=ser.count()-1) # two sided, so 1 - conf_level/2

    ci_low, ci_high = sample_mean - t_crit * sample_se,\
                     sample_mean + t_crit * sample_se
    
    return {
        f"mean of {name}": sample_mean,
        f"std of {name}": sample_std,
        f"se of {name}": sample_se,
        f"ci_low of {name}": ci_low,
        f"ci high of {name}": ci_high
    }

def compute_metrics(returns:dict[str:pd.DataFrame], rf):
    metrics = pd.DataFrame(columns=returns.keys())

    for name, ret in returns.items():
        cum_ret = (1+ret).cumprod()
        # mean returns
        dd = compute_from_simulation_level(ret.mean(), name="mean returns")

        # sharpe ratio
        dd.update(compute_from_simulation_level((ret.mean() - rf) / ret.std(), name="sharpe ratios"))

        # sortino ratio
        dd.update(compute_from_simulation_level((ret.mean() - rf) / np.maximum(0, -ret).std(), name="sortino ratios"))
        
        # average drawdown
        drawdown = 1 - (cum_ret / cum_ret.cummax())
        dd.update(compute_from_simulation_level(drawdown.mean(), name="average drawdowns"))

        # max drawdown
        dd.update(compute_from_simulation_level(drawdown.max(), name="max drawdowns"))

        # put all into dataframe
        metrics[name] = pd.Series(dd)

    """
    mean_returns = {}
    std_r = {}
    std_of_means = {}
    se_of_means = {}

    mean_sharpe_ratios = {}
    mean_sortino_ratios = {}

    for name, ret in returns.items():
        mean_returns[name] = np.mean(ret)
        std_r[name] = ret.std().mean()

        std_of_means[name] = ret.mean().std()
        se_of_means[name] = ret.mean().sem()

        mean_sharpe_ratios[name] = ((ret.mean() - rf) / ret.std()).mean()
        mean_sortino_ratios[name] = ((ret.mean() - rf) / np.maximum(0, -ret).std()).mean()


    metrics['Expected return'] = mean_returns
    metrics['Mean std'] = std_r

    metrics['std of mean returns'] = std_of_means
    metrics['se of mean returns'] = se_of_means

    metrics['Mean Sharpe Ratio'] = mean_sharpe_ratios
    metrics['Mean Sortino Ratio'] = mean_sortino_ratios
    """

    return metrics