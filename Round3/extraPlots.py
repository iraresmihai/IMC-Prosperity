"""
Extra diagnostics for the VEV voucher stat-arb.

Plots produced:
  1. BS-residual $ time series   per-voucher (market - BS_fit) in dollars
  2. Z-score time series         per-voucher rolling z of IV residual + entry/exit bands
  3. Delta and gamma             per-voucher over time (greeks driving the hedge)
  5. Smile residual heatmap      strike (y) x time (x), colored by IV - fit

Outputs:
  dollarResiduals.png    panel 1
  zScores.png            panel 2 (6 subplots)
  greeks.png             panel 3
  smileResidualHeatmap.png  panel 5

Conventions match ivSmile.py: price = avg of highest-volume best bid/ask.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from math import log, sqrt, erf, exp

# ---- Config ----
DAYS = [0, 1, 2]
DATA_DIR = 'ROUND_3Data'
SAMPLE_EVERY_N_TICKS = 50           # finer than ivSmile so time series look smooth
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

ROLLING_WINDOW = 1000               # ticks for z-score (matches backtest default)
ENTRY_Z = 2.0
EXIT_Z = 0.5


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


# ---- Black-Scholes ----
def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_pdf(x):
    return exp(-0.5 * x * x) / sqrt(2.0 * 3.141592653589793)


def bs_call(S, K, T, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0.0)
    v = sigma * sqrt(T)
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / v
    d2 = d1 - v
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def bs_greeks(S, K, T, sigma):
    if sigma <= 0 or T <= 0:
        delta = 1.0 if S > K else 0.0
        return delta, 0.0
    v = sigma * sqrt(T)
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / v
    delta = norm_cdf(d1)
    gamma = norm_pdf(d1) / (S * v)
    return delta, gamma


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


# ---- Build per-tick records ----
records = []
for day in DAYS:
    vev = price_series(UNDERLYING, day)
    voucher_ps = {name: price_series(name, day) for name, _ in VOUCHERS}
    tte_start = TTE_AT_START_DAYS[day]

    tss = vev.index.values[::SAMPLE_EVERY_N_TICKS]
    for ts in tss:
        S = float(vev.loc[ts])
        T = tte_start - ts / TICKS_PER_DAY_UNIT
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
df['t_abs'] = df['timestamp'] + df['day'] * TICKS_PER_DAY_UNIT
df = df.sort_values(['t_abs', 'strike']).reset_index(drop=True)
print(f'records: {len(df)}')


# ---- Per-tick smile fit on tradable strikes; compute fit IV, $ residual, greeks ----
fit_iv_col, dollar_resid_col, delta_col, gamma_col, iv_resid_col = [], [], [], [], []

for ts_abs, grp in df.groupby('t_abs', sort=False):
    fit_grp = grp[grp['strike_name'].isin(TRADABLE)]
    if len(fit_grp) >= 3:
        coeffs = np.polyfit(fit_grp['log_mny'], fit_grp['iv'], 2)
    else:
        coeffs = None
    for _, row in grp.iterrows():
        if coeffs is None:
            fit_iv_col.append(np.nan); dollar_resid_col.append(np.nan)
            delta_col.append(np.nan); gamma_col.append(np.nan); iv_resid_col.append(np.nan)
            continue
        iv_fit = float(np.polyval(coeffs, row['log_mny']))
        fit_iv_col.append(iv_fit)
        iv_resid_col.append(row['iv'] - iv_fit)
        bs_fit_price = bs_call(row['S'], float(row['strike']), row['TTE_days'], iv_fit)
        dollar_resid_col.append(row['C'] - bs_fit_price)
        d, g = bs_greeks(row['S'], float(row['strike']), row['TTE_days'], iv_fit)
        delta_col.append(d); gamma_col.append(g)

df['iv_fit'] = fit_iv_col
df['dollar_resid'] = dollar_resid_col
df['iv_resid'] = iv_resid_col
df['delta'] = delta_col
df['gamma'] = gamma_col


# ---- Plot 1: $ residual time series (tradable strikes) ----
fig1, ax1 = plt.subplots(figsize=(15, 6))
cmap = plt.colormaps['viridis']
for i, name in enumerate(TRADABLE):
    sub = df[df['strike_name'] == name].sort_values('t_abs')
    ax1.plot(sub['t_abs'], sub['dollar_resid'],
             color=cmap(i / max(1, len(TRADABLE) - 1)),
             linewidth=0.9, label=name)
ax1.axhline(0, color='black', linewidth=0.7)
for d in DAYS[1:]:
    ax1.axvline(d * TICKS_PER_DAY_UNIT, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)
ax1.set_xlabel('Timestamp (3 days concatenated)')
ax1.set_ylabel('Market - BS(fit IV)   ($)')
ax1.set_title('Dollar mispricing per voucher (market - BS price using smile-fit IV)')
ax1.grid(True, linewidth=0.3)
ax1.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False)
fig1.tight_layout()
fig1.savefig('dollarResiduals.png', dpi=140, bbox_inches='tight')
print('saved dollarResiduals.png')


# ---- Plot 2: z-score time series, 6-panel grid with entry/exit bands ----
fig2, axes2 = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
axes2 = axes2.ravel()
for i, name in enumerate(TRADABLE):
    ax = axes2[i]
    sub = df[df['strike_name'] == name].sort_values('t_abs').copy()
    r = sub['iv_resid']
    mu = r.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW // 4).mean()
    sd = r.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW // 4).std()
    z = (r - mu) / sd
    ax.plot(sub['t_abs'], z, color='steelblue', linewidth=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    for lvl, c in [(ENTRY_Z, 'red'), (-ENTRY_Z, 'red'), (EXIT_Z, 'green'), (-EXIT_Z, 'green')]:
        ax.axhline(lvl, color=c, linestyle='--', linewidth=0.7, alpha=0.7)
    for d in DAYS[1:]:
        ax.axvline(d * TICKS_PER_DAY_UNIT, color='grey', linestyle=':', linewidth=0.6, alpha=0.5)
    ax.set_title(f'{name}   z-score of IV residual', fontsize=10)
    ax.set_ylabel('z')
    ax.set_ylim(-5, 5)
    ax.grid(True, linewidth=0.3)
axes2[-1].set_xlabel('Timestamp')
axes2[-2].set_xlabel('Timestamp')
fig2.suptitle(f'Rolling z-score (window={ROLLING_WINDOW}). '
              f'red dashed = entry ±{ENTRY_Z}, green dashed = exit ±{EXIT_Z}',
              fontsize=12)
fig2.tight_layout(rect=[0, 0, 1, 0.97])
fig2.savefig('zScores.png', dpi=140, bbox_inches='tight')
print('saved zScores.png')


# ---- Plot 3: greeks over time (delta + gamma) for tradable strikes ----
fig3, (axd, axg) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
for i, name in enumerate(TRADABLE):
    sub = df[df['strike_name'] == name].sort_values('t_abs')
    color = cmap(i / max(1, len(TRADABLE) - 1))
    axd.plot(sub['t_abs'], sub['delta'], color=color, linewidth=0.9, label=name)
    axg.plot(sub['t_abs'], sub['gamma'], color=color, linewidth=0.9, label=name)
for d in DAYS[1:]:
    axd.axvline(d * TICKS_PER_DAY_UNIT, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)
    axg.axvline(d * TICKS_PER_DAY_UNIT, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)
axd.set_ylabel('Delta')
axd.set_title('Delta over time (per tradable voucher) — drives VEV hedge size')
axd.grid(True, linewidth=0.3)
axd.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False)
axg.set_ylabel('Gamma')
axg.set_xlabel('Timestamp')
axg.set_title('Gamma over time — hedge churn intensity')
axg.grid(True, linewidth=0.3)
fig3.tight_layout()
fig3.savefig('greeks.png', dpi=140, bbox_inches='tight')
print('saved greeks.png')


# ---- Plot 5: smile residual heatmap (strike x time) ----
heat = (df[df['strike_name'].isin(TRADABLE)]
        .pivot_table(index='strike', columns='t_abs', values='iv_resid', aggfunc='mean')
        .sort_index())
heat_vals = heat.values * 100        # %/day
vmax = np.nanpercentile(np.abs(heat_vals), 98)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

fig5, ax5 = plt.subplots(figsize=(15, 5))
im = ax5.imshow(heat_vals, aspect='auto', origin='lower', cmap='RdBu_r', norm=norm,
                extent=[heat.columns.min(), heat.columns.max(),
                        heat.index.min() - 50, heat.index.max() + 50])
ax5.set_yticks(heat.index.values)
ax5.set_yticklabels([str(int(k)) for k in heat.index.values])
ax5.set_xlabel('Timestamp (3 days concatenated)')
ax5.set_ylabel('Strike')
ax5.set_title('Smile residual heatmap  (IV - fit, %/day).  red = rich, blue = cheap')
for d in DAYS[1:]:
    ax5.axvline(d * TICKS_PER_DAY_UNIT, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
fig5.colorbar(im, ax=ax5, label='IV - fit (%/day)')
fig5.tight_layout()
fig5.savefig('smileResidualHeatmap.png', dpi=140, bbox_inches='tight')
print('saved smileResidualHeatmap.png')


# ---- Quick console summary ----
print('\n$ residual stats per tradable voucher:')
stats = (df[df['strike_name'].isin(TRADABLE)]
         .groupby('strike_name')['dollar_resid']
         .agg(['mean', 'std', 'min', 'max']).round(3))
print(stats.to_string())
