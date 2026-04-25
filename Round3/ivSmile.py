"""
Implied volatility smile for VEV vouchers.

For every (subsampled) tick, we invert Black-Scholes per voucher to get its
implied volatility, then plot IV vs log-moneyness log(K/S).  A quadratic fit
through the tradable strikes gives the smile shape; residuals from that fit
are the per-voucher mispricing signal.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt, exp, erf

# ---- Config ----
DAYS = [0, 1, 2]
DATA_DIR = 'ROUND_3Data'
SAMPLE_EVERY_N_TICKS = 100       # keep 1 out of every N ticks for scatter
TICKS_PER_DAY_UNIT = 1_000_000   # timestamp step across a day
TTE_AT_START_DAYS = {0: 8.0, 1: 7.0, 2: 6.0}

VOUCHERS = [
    ('VEV_4000', 4000), ('VEV_4500', 4500), ('VEV_5000', 5000),
    ('VEV_5100', 5100), ('VEV_5200', 5200), ('VEV_5300', 5300),
    ('VEV_5400', 5400), ('VEV_5500', 5500), ('VEV_6000', 6000),
    ('VEV_6500', 6500),
]
UNDERLYING = 'VELVETFRUIT_EXTRACT'


# ---- Price helpers (price = avg of highest-volume best bid/ask) ----
def best_by_volume(df, side):
    rows = []
    for lvl in [1, 2, 3]:
        tmp = df[['timestamp', f'{side}_price_{lvl}', f'{side}_volume_{lvl}']].copy()
        tmp.columns = ['timestamp', 'price', 'volume']
        rows.append(tmp)
    stacked = pd.concat(rows).dropna()
    idx = stacked.groupby('timestamp')['volume'].idxmax()
    out = stacked.loc[idx, ['timestamp', 'price']].sort_values('timestamp')
    return out.drop_duplicates(subset='timestamp').reset_index(drop=True)


_day_cache = {}
def day_frame(d):
    if d not in _day_cache:
        _day_cache[d] = pd.read_csv(f'{DATA_DIR}/prices_round_3_day_{d}.csv', sep=';')
    return _day_cache[d]


def price_series(product, day):
    df = day_frame(day)
    sub = df[df['product'] == product]
    if sub.empty:
        return pd.Series(dtype=float)
    bb = best_by_volume(sub, 'bid').rename(columns={'price': 'bid'})
    ba = best_by_volume(sub, 'ask').rename(columns={'price': 'ask'})
    m = bb.merge(ba, on='timestamp', how='inner')
    m['mid'] = (m['bid'] + m['ask']) / 2
    return m.set_index('timestamp')['mid']


# ---- Black-Scholes pricer + bisection inverter ----
def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call(S, K, T, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0.0)
    v = sigma * sqrt(T)
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / v
    d2 = d1 - v
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def implied_vol(C, S, K, T, lo=1e-5, hi=5.0, tol=1e-6, max_iter=80):
    intrinsic = max(S - K, 0.0)
    if T <= 0 or C <= intrinsic + 1e-6 or C >= S:
        return float('nan')
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = bs_call(S, K, T, mid)
        if p < C:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---- Compute IV records ----
records = []
for day in DAYS:
    vev = price_series(UNDERLYING, day)
    voucher_ps = {name: price_series(name, day) for name, _ in VOUCHERS}
    tte_start = TTE_AT_START_DAYS[day]

    tss = vev.index.values[::SAMPLE_EVERY_N_TICKS]
    for ts in tss:
        S = float(vev.loc[ts])
        T = tte_start - ts / TICKS_PER_DAY_UNIT       # days
        if T <= 0:
            continue
        for name, K in VOUCHERS:
            ps = voucher_ps[name]
            if ts not in ps.index:
                continue
            C = float(ps.loc[ts])
            iv = implied_vol(C, S, float(K), T)
            if np.isnan(iv):
                continue
            records.append({
                'day': day, 'timestamp': ts, 'strike': K, 'strike_name': name,
                'S': S, 'C': C, 'TTE_days': T, 'iv': iv,
                'log_mny': log(K / S),
            })

df = pd.DataFrame(records)
print(f'IV observations computed: {len(df)}')
print('Per-voucher IV summary (per-day sigma, in %):')
summary = df.groupby('strike_name')['iv'].agg(['count', 'mean', 'std', 'min', 'max'])
summary[['mean', 'std', 'min', 'max']] *= 100
print(summary.round(3).to_string())

# Strikes to fit the smile on (exclude deep ITM at intrinsic + floor-pinned OTM).
FIT_STRIKES = {'VEV_5000', 'VEV_5100', 'VEV_5200', 'VEV_5300', 'VEV_5400', 'VEV_5500'}


# ---- Plot: smile scatter with per-day parabolic fit (fit only on tradable strikes) ----
fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.22, height_ratios=[1.3, 1, 1])

ax_smile = fig.add_subplot(gs[0, :])
ax_ts = fig.add_subplot(gs[1, :])
ax_res = fig.add_subplot(gs[2, :])

day_colors = {0: '#2ca02c', 1: '#d62728', 2: '#9467bd'}

for day in DAYS:
    sub = df[df['day'] == day]
    if sub.empty:
        continue
    fit_sub = sub[sub['strike_name'].isin(FIT_STRIKES)]
    non_fit = sub[~sub['strike_name'].isin(FIT_STRIKES)]

    # dots: tradable strikes solid, excluded strikes grey and faint
    ax_smile.scatter(fit_sub['log_mny'], fit_sub['iv'] * 100,
                     s=14, alpha=0.5, color=day_colors[day],
                     label=f'day {day} (fit band)')
    ax_smile.scatter(non_fit['log_mny'], non_fit['iv'] * 100,
                     s=8, alpha=0.15, color=day_colors[day],
                     marker='x')

    # parabolic fit on tradable strikes only
    coeffs = np.polyfit(fit_sub['log_mny'], fit_sub['iv'], 2)
    xs = np.linspace(fit_sub['log_mny'].min(), fit_sub['log_mny'].max(), 150)
    ax_smile.plot(xs, np.polyval(coeffs, xs) * 100,
                  color=day_colors[day], linewidth=2.0,
                  label=f'day {day} fit: {coeffs[0]:.2f}·m² {coeffs[1]:+.3f}·m {coeffs[2]:+.5f}')

ax_smile.axvline(0, color='grey', linestyle=':', linewidth=0.7)
ax_smile.set_xlabel('log-moneyness  m = log(K / S)   ←  ITM    |    ATM    |    OTM  →')
ax_smile.set_ylabel('Implied vol  (per-day σ, %)')
ax_smile.set_title('Implied volatility smile — IV vs log(K/S), parabolic fit on K ∈ [5000, 5500]')
ax_smile.grid(True, linewidth=0.3)
ax_smile.legend(loc='upper left', fontsize=8, ncol=2)
# Zoom y-axis on the meaningful range
fit_ivs = df[df['strike_name'].isin(FIT_STRIKES)]['iv'].values * 100
if len(fit_ivs):
    ax_smile.set_ylim(fit_ivs.min() - 0.1, fit_ivs.max() + 0.2)


# ---- IV time series (tradable strikes only) ----
cmap = plt.colormaps['viridis']
tradable = [n for n, _ in VOUCHERS if n in FIT_STRIKES and n in df['strike_name'].unique()]
for i, name in enumerate(tradable):
    sub = df[df['strike_name'] == name].copy()
    sub['t_abs'] = sub['timestamp'] + sub['day'] * TICKS_PER_DAY_UNIT
    sub = sub.sort_values('t_abs')
    ax_ts.plot(sub['t_abs'], sub['iv'] * 100,
               color=cmap(i / max(1, len(tradable) - 1)),
               linewidth=0.9, label=name)

for d in DAYS[1:]:
    ax_ts.axvline(d * TICKS_PER_DAY_UNIT, color='black', linestyle='--', linewidth=0.7, alpha=0.5)
ax_ts.set_xlabel('Timestamp (3 days concatenated)')
ax_ts.set_ylabel('Implied vol (per-day σ, %)')
ax_ts.set_title('Per-voucher IV over time (tradable strikes only)')
ax_ts.grid(True, linewidth=0.3)
ax_ts.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)


# ---- Residuals: per-tick residuals for the tradable strikes only ----
if not df.empty:
    fit_df = df[df['strike_name'].isin(FIT_STRIKES)].copy()
    coeffs_all = np.polyfit(fit_df['log_mny'], fit_df['iv'], 2)

    # Apply fit to ALL vouchers (tradable + excluded) so we can see who's off
    df['iv_fit'] = np.polyval(coeffs_all, df['log_mny'])
    df['resid'] = df['iv'] - df['iv_fit']

    tradable_resid = df[df['strike_name'].isin(FIT_STRIKES)]
    resid_stats = (tradable_resid.groupby('strike_name')
                                 .agg(mean_resid=('resid', 'mean'),
                                      std_resid=('resid', 'std'),
                                      n=('resid', 'size'))
                                 .reindex([n for n, _ in VOUCHERS if n in FIT_STRIKES]))

    xs = np.arange(len(resid_stats))
    ax_res.bar(xs, resid_stats['mean_resid'] * 100,
               yerr=resid_stats['std_resid'] * 100, capsize=4,
               color=['#2ca02c' if v > 0 else '#d62728' for v in resid_stats['mean_resid']],
               alpha=0.75, edgecolor='black', linewidth=0.6)
    ax_res.axhline(0, color='grey', linewidth=0.7)
    ax_res.set_xticks(xs)
    ax_res.set_xticklabels(resid_stats.index, fontsize=9)
    ax_res.set_ylabel('mean residual  (IV - fit, %/day)')
    ax_res.set_title('Per-strike mean residual vs pooled parabolic smile  (error bars = tick-by-tick std)\n'
                     'green = voucher tends to trade rich  •  red = trades cheap')
    ax_res.grid(True, linewidth=0.3, axis='y')

    print(f'\nPooled smile fit (on tradable strikes, across all 3 days):')
    print(f'  sigma(m) = {coeffs_all[0]:.4f} * m^2 + {coeffs_all[1]:+.4f} * m + {coeffs_all[2]:+.6f}')
    print('\nPer-strike residuals (IV - fit), in %/day (tradable strikes):')
    print((resid_stats[['mean_resid', 'std_resid']] * 100).round(4).to_string())

fig.suptitle('Implied volatility analysis — VEV vouchers', fontsize=13, y=0.995)
fig.savefig('ivSmile.png', dpi=140, bbox_inches='tight')
print('\nsaved ivSmile.png')
