"""
Same diagnostics as extraPlots.py, but with a FLAT IV benchmark instead of a
parabolic smile fit.

The hypothesis (from the Magritte hint): IMC's hidden fair value uses ONE sigma
across all strikes. The market smile is an artifact, not a real surface, so
fitting a parabola absorbs real mispricing into the curve. Using a single sigma
keeps the full mispricing visible as a tradable residual.

Flat-IV choices (FLAT_IV_MODE):
  'per_tick_mean'  per-tick mean IV across tradable strikes (tracks base level)
  'pooled_mean'    one constant = pooled mean across all ticks/strikes
  'fixed'          one constant supplied via FLAT_IV_FIXED (per-day, e.g. 0.0125)

Outputs:
  dollarResiduals_flat.png
  zScores_flat.png
  greeks_flat.png
  ivDeviationHeatmap_flat.png      (IV - flat_iv) per strike over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from math import log, sqrt, erf, exp

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

FLAT_IV_MODE = 'fixed'              # 'per_tick_mean' | 'pooled_mean' | 'fixed'
FLAT_IV_FIXED = 0.01262             # 20% annualized (~1.262%/day); IMC's likely hidden sigma

ROLLING_WINDOW = 1000
ENTRY_Z = 2.0
EXIT_Z = 0.5


# ---- Price helpers ----
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
    return norm_cdf(d1), norm_pdf(d1) / (S * v)


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


# ---- Build records ----
records = []
for day in DAYS:
    vev = price_series(UNDERLYING, day)
    voucher_ps = {name: price_series(name, day) for name, _ in VOUCHERS}
    tte_start = TTE_AT_START_DAYS[day]

    for ts in vev.index.values[::SAMPLE_EVERY_N_TICKS]:
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


# ---- Pick the flat IV ----
trad_df = df[df['strike_name'].isin(TRADABLE)]

if FLAT_IV_MODE == 'pooled_mean':
    flat_iv_const = trad_df['iv'].mean()
    flat_iv_per_t = pd.Series(flat_iv_const, index=df['t_abs'].unique())
    flat_label = f'pooled mean = {flat_iv_const*100:.3f}%/day'
elif FLAT_IV_MODE == 'fixed':
    flat_iv_const = FLAT_IV_FIXED
    flat_iv_per_t = pd.Series(flat_iv_const, index=df['t_abs'].unique())
    flat_label = f'fixed = {flat_iv_const*100:.3f}%/day'
elif FLAT_IV_MODE == 'per_tick_mean':
    flat_iv_per_t = trad_df.groupby('t_abs')['iv'].mean()
    flat_iv_const = float(flat_iv_per_t.mean())
    flat_label = f'per-tick mean across tradable strikes (avg = {flat_iv_const*100:.3f}%/day)'
else:
    raise ValueError(FLAT_IV_MODE)

df['flat_iv'] = df['t_abs'].map(flat_iv_per_t)
df['iv_resid'] = df['iv'] - df['flat_iv']

# BS price at flat IV, dollar residual, greeks at flat IV
def row_apply(r):
    bs_fair = bs_call(r['S'], float(r['strike']), r['TTE_days'], r['flat_iv'])
    d, g = bs_greeks(r['S'], float(r['strike']), r['TTE_days'], r['flat_iv'])
    return pd.Series({'bs_flat': bs_fair, 'dollar_resid': r['C'] - bs_fair,
                      'delta': d, 'gamma': g})

df[['bs_flat', 'dollar_resid', 'delta', 'gamma']] = df.apply(row_apply, axis=1)


# ---- Plot 1: $ residual under flat IV ----
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
ax1.set_ylabel('Market - BS(flat IV)   ($)')
ax1.set_title(f'Dollar mispricing per voucher  vs FLAT IV  [{flat_label}]')
ax1.grid(True, linewidth=0.3)
ax1.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False)
fig1.tight_layout()
fig1.savefig('dollarResiduals_flat.png', dpi=140, bbox_inches='tight')
print('saved dollarResiduals_flat.png')


# ---- Plot 2: z-scores under flat IV ----
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
    ax.set_title(f'{name}  z of (IV - flat_IV)', fontsize=10)
    ax.set_ylabel('z')
    ax.set_ylim(-5, 5)
    ax.grid(True, linewidth=0.3)
axes2[-1].set_xlabel('Timestamp')
axes2[-2].set_xlabel('Timestamp')
fig2.suptitle(f'Rolling z-score of IV deviation from flat IV  '
              f'(window={ROLLING_WINDOW}, entry ±{ENTRY_Z}, exit ±{EXIT_Z})',
              fontsize=12)
fig2.tight_layout(rect=[0, 0, 1, 0.97])
fig2.savefig('zScores_flat.png', dpi=140, bbox_inches='tight')
print('saved zScores_flat.png')


# ---- Plot 3: greeks at flat IV ----
fig3, (axd, axg) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
for i, name in enumerate(TRADABLE):
    sub = df[df['strike_name'] == name].sort_values('t_abs')
    color = cmap(i / max(1, len(TRADABLE) - 1))
    axd.plot(sub['t_abs'], sub['delta'], color=color, linewidth=0.9, label=name)
    axg.plot(sub['t_abs'], sub['gamma'], color=color, linewidth=0.9, label=name)
for d in DAYS[1:]:
    axd.axvline(d * TICKS_PER_DAY_UNIT, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)
    axg.axvline(d * TICKS_PER_DAY_UNIT, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)
axd.set_ylabel('Delta'); axd.grid(True, linewidth=0.3)
axd.set_title(f'Delta at flat IV  [{flat_label}] — drives VEV hedge size')
axd.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False)
axg.set_ylabel('Gamma'); axg.set_xlabel('Timestamp'); axg.grid(True, linewidth=0.3)
axg.set_title('Gamma at flat IV — hedge churn intensity')
fig3.tight_layout()
fig3.savefig('greeks_flat.png', dpi=140, bbox_inches='tight')
print('saved greeks_flat.png')


# ---- Plot 5: heatmap of (IV - flat_IV) ----
heat = (df[df['strike_name'].isin(TRADABLE)]
        .pivot_table(index='strike', columns='t_abs', values='iv_resid', aggfunc='mean')
        .sort_index())
heat_vals = heat.values * 100
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
ax5.set_title(f'IV deviation from flat IV  ({flat_label})  red = rich, blue = cheap')
for d in DAYS[1:]:
    ax5.axvline(d * TICKS_PER_DAY_UNIT, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
fig5.colorbar(im, ax=ax5, label='IV - flat_IV (%/day)')
fig5.tight_layout()
fig5.savefig('ivDeviationHeatmap_flat.png', dpi=140, bbox_inches='tight')
print('saved ivDeviationHeatmap_flat.png')


# ---- Console summary ----
print(f'\nFlat IV mode: {FLAT_IV_MODE}   ({flat_label})')
print('\n$ residual stats per tradable voucher (vs flat IV):')
stats = (df[df['strike_name'].isin(TRADABLE)]
         .groupby('strike_name')['dollar_resid']
         .agg(['mean', 'std', 'min', 'max']).round(3))
print(stats.to_string())
