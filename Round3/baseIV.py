"""
Base IV time series: per-tick mean implied vol across tradable strikes.

Question: is the flat IV stable enough to treat as a single fixed sigma, or is
it a slowly-varying state we need to track?

Plots:
  Top:    base IV over time (per-tick mean across K=5000..5500)
          + rolling mean and pooled mean for reference
  Mid:    histogram of base IV values (3 days pooled)
  Bottom: per-day distribution + summary stats table
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt, erf

# ---- Config ----
DAYS = [0, 1, 2]
DATA_DIR = 'ROUND_3Data'
SAMPLE_EVERY_N_TICKS = 50
TICKS_PER_DAY_UNIT = 1_000_000
TTE_AT_START_DAYS = {0: 8.0, 1: 7.0, 2: 6.0}

VOUCHERS = [
    ('VEV_4000', 4000), ('VEV_4500', 4500), ('VEV_5000', 5000),
    ('VEV_5100', 5100), ('VEV_5200', 5200), ('VEV_5300', 5300),
    ('VEV_5400', 5400), ('VEV_5500', 5500), ('VEV_6000', 6000),
    ('VEV_6500', 6500),
]
TRADABLE = ['VEV_5000', 'VEV_5100', 'VEV_5200', 'VEV_5300', 'VEV_5400', 'VEV_5500']
UNDERLYING = 'VELVETFRUIT_EXTRACT'

ROLLING_WINDOW = 500


# ---- Helpers (same conventions) ----
def best_by_volume(df, side):
    rows = []
    for lvl in [1, 2, 3]:
        tmp = df[['timestamp', f'{side}_price_{lvl}', f'{side}_volume_{lvl}']].copy()
        tmp.columns = ['timestamp', 'price', 'volume']
        rows.append(tmp)
    stacked = pd.concat(rows).dropna()
    idx = stacked.groupby('timestamp')['volume'].idxmax()
    return (stacked.loc[idx, ['timestamp', 'price']]
            .sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True))


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
        if bs_call(S, K, T, mid) < C:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---- Compute IVs ----
records = []
for day in DAYS:
    vev = price_series(UNDERLYING, day)
    voucher_ps = {n: price_series(n, day) for n, _ in VOUCHERS if n in TRADABLE}
    tte_start = TTE_AT_START_DAYS[day]
    for ts in vev.index.values[::SAMPLE_EVERY_N_TICKS]:
        S = float(vev.loc[ts])
        T = tte_start - ts / TICKS_PER_DAY_UNIT
        if T <= 0:
            continue
        for name in TRADABLE:
            ps = voucher_ps[name]
            if ts not in ps.index:
                continue
            K = float(name.split('_')[1])
            iv = implied_vol(float(ps.loc[ts]), S, K, T)
            if not np.isnan(iv):
                records.append({'day': day, 'timestamp': ts, 'iv': iv,
                                't_abs': ts + day * TICKS_PER_DAY_UNIT})

df = pd.DataFrame(records)
base = df.groupby('t_abs')['iv'].mean().sort_index()
base_pct = base * 100      # plot in %/day


# ---- Stats ----
pooled_mean = base_pct.mean()
pooled_std = base_pct.std()
per_day = (df.assign(day_idx=lambda d: d['t_abs'] // TICKS_PER_DAY_UNIT)
             .groupby('day_idx')['iv']
             .agg(['mean', 'std', 'min', 'max'])
             .rename_axis('day') * 100)

print(f'Base IV pooled mean: {pooled_mean:.4f}%/day   std: {pooled_std:.4f}%/day')
print(f'Coefficient of variation: {pooled_std / pooled_mean * 100:.2f}%')
print('\nPer-day stats:')
print(per_day.round(4).to_string())


# ---- Plot ----
fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(3, 1, height_ratios=[1.6, 1, 1], hspace=0.35)

ax_ts = fig.add_subplot(gs[0])
ax_hist = fig.add_subplot(gs[1])
ax_per = fig.add_subplot(gs[2])

# top: time series with rolling mean and pooled mean
ax_ts.plot(base_pct.index, base_pct.values, color='steelblue', linewidth=0.7,
           alpha=0.8, label='base IV (per-tick mean across tradable strikes)')
roll = base_pct.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW // 4).mean()
ax_ts.plot(roll.index, roll.values, color='darkorange', linewidth=1.8,
           label=f'rolling mean (window={ROLLING_WINDOW})')
ax_ts.axhline(pooled_mean, color='red', linestyle='--', linewidth=1.2,
              label=f'pooled mean = {pooled_mean:.3f}%/day')
ax_ts.fill_between(base_pct.index, pooled_mean - pooled_std, pooled_mean + pooled_std,
                   color='red', alpha=0.08, label='±1σ band (pooled)')
for d in DAYS[1:]:
    ax_ts.axvline(d * TICKS_PER_DAY_UNIT, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)
ax_ts.set_xlabel('Timestamp (3 days concatenated)')
ax_ts.set_ylabel('Base IV (%/day)')
ax_ts.set_title('Base IV over time — per-tick mean implied vol across tradable strikes (K=5000..5500)')
ax_ts.legend(loc='upper right', fontsize=9)
ax_ts.grid(True, linewidth=0.3)

# middle: histogram pooled
ax_hist.hist(base_pct.values, bins=80, color='steelblue', edgecolor='black',
             linewidth=0.4, alpha=0.85)
ax_hist.axvline(pooled_mean, color='red', linestyle='--', linewidth=1.5,
                label=f'mean = {pooled_mean:.3f}%/day')
ax_hist.axvline(pooled_mean - pooled_std, color='red', linestyle=':', linewidth=1)
ax_hist.axvline(pooled_mean + pooled_std, color='red', linestyle=':', linewidth=1)
ax_hist.set_xlabel('Base IV (%/day)')
ax_hist.set_ylabel('count')
ax_hist.set_title(f'Distribution of base IV  (3 days pooled).  std = {pooled_std:.4f}%/day, '
                  f'CV = {pooled_std / pooled_mean * 100:.1f}%')
ax_hist.legend(fontsize=9)
ax_hist.grid(True, linewidth=0.3, axis='y')

# bottom: per-day box-style overlay (separate hists)
day_colors = {0: '#2ca02c', 1: '#d62728', 2: '#9467bd'}
for d in DAYS:
    sub = base_pct[(base_pct.index >= d * TICKS_PER_DAY_UNIT) &
                   (base_pct.index < (d + 1) * TICKS_PER_DAY_UNIT)]
    ax_per.hist(sub.values, bins=60, alpha=0.45, color=day_colors[d],
                edgecolor='black', linewidth=0.3,
                label=f'day {d}  (μ={sub.mean():.3f}, σ={sub.std():.3f})')
ax_per.set_xlabel('Base IV (%/day)')
ax_per.set_ylabel('count')
ax_per.set_title('Base IV distribution by day — drift between days?')
ax_per.legend(fontsize=9)
ax_per.grid(True, linewidth=0.3, axis='y')

fig.savefig('baseIV.png', dpi=140, bbox_inches='tight')
print('saved baseIV.png')
