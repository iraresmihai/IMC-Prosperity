"""
Backtest a volatility stat-arb strategy on one VEV voucher.

Logic per tick:
  1) S = mid(VEV)  [mid = avg of highest-volume best bid/ask]
  2) Fit quadratic IV smile on K in FIT_STRIKES using BS inversion.
  3) For the target VOUCHER, compute residual = iv_market - iv_fit.
  4) Rolling z-score of residual over ROLLING_WINDOW ticks.
  5) Position rules:
        z < -ENTRY_Z              -> long  +POSITION_LIMIT (voucher cheap)
        z > +ENTRY_Z              -> short -POSITION_LIMIT (voucher rich)
        |z| < EXIT_Z              -> flat
        otherwise                 -> hold current position
  6) Optional delta hedge: short (delta * voucher_pos) of VEV.
  7) MTM PnL = cash + voucher_pos * voucher_mid + vev_pos * vev_mid.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt, erf

# =================== USER CONFIG ===================
VOUCHER          = 'VEV_5400'   # which voucher to trade
ENTRY_Z          = 2.0          # |z| to open
EXIT_Z           = 0.5          # |z| to close
ROLLING_WINDOW   = 1000         # ticks for rolling mu/sigma of residual
POSITION_LIMIT   = 300          # voucher position limit
VEV_LIMIT        = 200          # underlying position limit
DELTA_HEDGE      = True         # hedge voucher position with VEV
CROSS_SPREAD     = True        # True = buy@ask / sell@bid; False = trade at mid
DAYS             = [0, 1, 2]
# ===================================================

DATA_DIR    = 'ROUND_3Data'
UNDERLYING  = 'VELVETFRUIT_EXTRACT'
VOUCHERS_ALL = [
    ('VEV_4000', 4000), ('VEV_4500', 4500), ('VEV_5000', 5000),
    ('VEV_5100', 5100), ('VEV_5200', 5200), ('VEV_5300', 5300),
    ('VEV_5400', 5400), ('VEV_5500', 5500), ('VEV_6000', 6000),
    ('VEV_6500', 6500),
]
STRIKE_MAP   = dict(VOUCHERS_ALL)
FIT_STRIKES  = ['VEV_5000', 'VEV_5100', 'VEV_5200', 'VEV_5300', 'VEV_5400', 'VEV_5500']
TTE_START    = {0: 8.0, 1: 7.0, 2: 6.0}  # days at start of each historical day
K_TRADE      = STRIKE_MAP[VOUCHER]


# ---- Price helpers ----
def best_by_vol(df, side):
    rows = []
    for lvl in [1, 2, 3]:
        tmp = df[['timestamp', f'{side}_price_{lvl}', f'{side}_volume_{lvl}']].copy()
        tmp.columns = ['timestamp', 'price', 'volume']
        rows.append(tmp)
    st = pd.concat(rows).dropna()
    idx = st.groupby('timestamp')['volume'].idxmax()
    out = st.loc[idx, ['timestamp', 'price']].sort_values('timestamp')
    return out.drop_duplicates(subset='timestamp').reset_index(drop=True)


def prod_frame(df, product):
    sub = df[df['product'] == product]
    if sub.empty:
        return pd.DataFrame()
    bb = best_by_vol(sub, 'bid').rename(columns={'price': 'bid'})
    ba = best_by_vol(sub, 'ask').rename(columns={'price': 'ask'})
    m = bb.merge(ba, on='timestamp', how='inner')
    m['mid'] = (m['bid'] + m['ask']) / 2
    return m.set_index('timestamp')


# ---- Black-Scholes ----
def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call(S, K, T, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0.0)
    v = sigma * sqrt(T)
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / v
    d2 = d1 - v
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def bs_delta(S, K, T, sigma):
    if sigma <= 0 or T <= 0:
        return 1.0 if S > K else 0.0
    v = sigma * sqrt(T)
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / v
    return norm_cdf(d1)


def implied_vol(C, S, K, T):
    intr = max(S - K, 0.0)
    if T <= 0 or C <= intr + 1e-6 or C >= S:
        return float('nan')
    lo, hi = 1e-5, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_call(S, K, T, mid) < C:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7:
            break
    return 0.5 * (lo + hi)


# ---- Load data ----
print(f'Loading days {DAYS}...')
day_data = {}
for d in DAYS:
    raw = pd.read_csv(f'{DATA_DIR}/prices_round_3_day_{d}.csv', sep=';')
    per_prod = {p: prod_frame(raw, p) for p in STRIKE_MAP.keys()}
    per_prod[UNDERLYING] = prod_frame(raw, UNDERLYING)
    day_data[d] = per_prod


# ---- Build per-tick signal ----
print('Computing per-tick IV smile and residuals...')
rows = []
for d in DAYS:
    vev_df = day_data[d][UNDERLYING]
    voucher_df = day_data[d][VOUCHER]
    fit_dfs = {p: day_data[d][p] for p in FIT_STRIKES}

    for ts in vev_df.index:
        if ts not in voucher_df.index:
            continue
        S = float(vev_df.loc[ts, 'mid'])
        T = TTE_START[d] - ts / 1_000_000
        if T <= 0:
            continue

        ms, ivs = [], []
        for p in FIT_STRIKES:
            dfp = fit_dfs[p]
            if ts not in dfp.index:
                continue
            Kp = STRIKE_MAP[p]
            ivp = implied_vol(float(dfp.loc[ts, 'mid']), S, Kp, T)
            if not np.isnan(ivp):
                ms.append(log(Kp / S))
                ivs.append(ivp)
        if len(ivs) < 4:
            continue
        coeffs = np.polyfit(ms, ivs, 2)

        C_t = float(voucher_df.loc[ts, 'mid'])
        iv_mkt = implied_vol(C_t, S, K_TRADE, T)
        if np.isnan(iv_mkt):
            continue
        iv_fit = float(np.polyval(coeffs, log(K_TRADE / S)))

        rows.append({
            't_abs': ts + d * 1_000_000, 'day': d, 'ts': ts,
            'S': S, 'T': T,
            'v_mid': C_t,
            'v_bid': float(voucher_df.loc[ts, 'bid']),
            'v_ask': float(voucher_df.loc[ts, 'ask']),
            'vev_mid': S,
            'vev_bid': float(vev_df.loc[ts, 'bid']),
            'vev_ask': float(vev_df.loc[ts, 'ask']),
            'iv_mkt': iv_mkt, 'iv_fit': iv_fit,
            'resid': iv_mkt - iv_fit,
            'delta': bs_delta(S, K_TRADE, T, iv_fit),
        })

sig = pd.DataFrame(rows).sort_values('t_abs').reset_index(drop=True)
print(f'  signal ticks: {len(sig)}')

sig['mu']    = sig['resid'].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
sig['sigma'] = sig['resid'].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std()
sig['z']     = (sig['resid'] - sig['mu']) / sig['sigma']


# ---- Strategy loop ----
print('Running strategy...')
v_pos, vev_pos, cash = 0, 0, 0.0
n_trades_v, n_trades_vev = 0, 0

v_pos_arr   = np.zeros(len(sig), dtype=int)
vev_pos_arr = np.zeros(len(sig), dtype=int)
pnl_arr     = np.zeros(len(sig))

zs      = sig['z'].to_numpy()
deltas  = sig['delta'].to_numpy()
v_mid   = sig['v_mid'].to_numpy()
v_bid   = sig['v_bid'].to_numpy()
v_ask   = sig['v_ask'].to_numpy()
vev_mid = sig['vev_mid'].to_numpy()
vev_bid = sig['vev_bid'].to_numpy()
vev_ask = sig['vev_ask'].to_numpy()

for i in range(len(sig)):
    z = zs[i]

    # target voucher position (stateful hysteresis)
    if np.isnan(z):
        target = v_pos
    elif v_pos == 0:
        if   z < -ENTRY_Z: target =  POSITION_LIMIT
        elif z >  ENTRY_Z: target = -POSITION_LIMIT
        else: target = 0
    elif v_pos > 0:
        if   z >  ENTRY_Z:  target = -POSITION_LIMIT   # flip
        elif z > -EXIT_Z:   target = 0                 # close
        else: target = v_pos
    else:  # short
        if   z < -ENTRY_Z:  target =  POSITION_LIMIT   # flip
        elif z <  EXIT_Z:   target = 0
        else: target = v_pos

    qty = target - v_pos
    if qty != 0:
        px = (v_ask[i] if qty > 0 else v_bid[i]) if CROSS_SPREAD else v_mid[i]
        cash -= qty * px
        v_pos = target
        n_trades_v += 1

    if DELTA_HEDGE:
        tgt_vev = int(round(-v_pos * deltas[i]))
        tgt_vev = max(-VEV_LIMIT, min(VEV_LIMIT, tgt_vev))
        dv = tgt_vev - vev_pos
        if dv != 0:
            pvev = (vev_ask[i] if dv > 0 else vev_bid[i]) if CROSS_SPREAD else vev_mid[i]
            cash -= dv * pvev
            vev_pos = tgt_vev
            n_trades_vev += 1

    mtm = cash + v_pos * v_mid[i] + vev_pos * vev_mid[i]
    v_pos_arr[i] = v_pos
    vev_pos_arr[i] = vev_pos
    pnl_arr[i] = mtm

sig['v_pos']   = v_pos_arr
sig['vev_pos'] = vev_pos_arr
sig['pnl']     = pnl_arr


# ---- Summary ----
peak = np.maximum.accumulate(pnl_arr)
dd = pnl_arr - peak

print('\n' + '=' * 50)
print('BACKTEST RESULTS')
print('=' * 50)
print(f'Voucher:           {VOUCHER}  (K = {K_TRADE})')
print(f'Entry / Exit z:    ±{ENTRY_Z} / ±{EXIT_Z}')
print(f'Rolling window:    {ROLLING_WINDOW} ticks')
print(f'Delta hedge:       {DELTA_HEDGE}')
print(f'Cross spread:      {CROSS_SPREAD}')
print(f'Days:              {DAYS}')
print(f'Signal ticks:      {len(sig)}')
print(f'Voucher trades:    {n_trades_v}')
print(f'VEV hedge trades:  {n_trades_vev}')
print(f'FINAL PnL:         {pnl_arr[-1]:+.2f}')
print(f'Max drawdown:      {dd.min():.2f}')
for d in DAYS:
    sub = sig[sig['day'] == d]
    if sub.empty: continue
    start = sub['pnl'].iloc[0] - (sub['v_pos'].iloc[0] * sub['v_mid'].iloc[0]
                                   + sub['vev_pos'].iloc[0] * sub['vev_mid'].iloc[0]
                                   - (sub['pnl'].iloc[0]))   # start-of-day PnL
    # simpler: PnL at end of day minus PnL just before this day started
    end_pnl = sub['pnl'].iloc[-1]
    prev_end = sig[sig['day'] < d]['pnl'].iloc[-1] if d > DAYS[0] else 0.0
    print(f'  day {d}:            {end_pnl - prev_end:+.2f}')
print('=' * 50)


# ---- Plot ----
fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True,
                         gridspec_kw={'height_ratios': [1.2, 1, 1, 1.2]})

ax = axes[0]
ax.plot(sig['t_abs'], sig['v_mid'], color='#1f77b4', linewidth=0.7, label=f'{VOUCHER} mid')
ax.set_ylabel(f'{VOUCHER} mid')
ax.grid(True, linewidth=0.3)
ax.set_title(f'Backtest — {VOUCHER}  |  entry ±{ENTRY_Z}σ / exit ±{EXIT_Z}σ  |  window {ROLLING_WINDOW}  |  delta_hedge={DELTA_HEDGE}')
ax.legend(loc='upper right', fontsize=8)

ax = axes[1]
ax.plot(sig['t_abs'], sig['z'], color='#ff7f0e', linewidth=0.6)
for lv, ls, c in [(ENTRY_Z, '--', '#2ca02c'), (-ENTRY_Z, '--', '#2ca02c'),
                   (EXIT_Z, ':', '#808080'), (-EXIT_Z, ':', '#808080')]:
    ax.axhline(lv, color=c, linestyle=ls, linewidth=0.8)
ax.axhline(0, color='black', linewidth=0.4)
ax.set_ylabel('z-score')
ax.set_ylim(-5, 5)
ax.grid(True, linewidth=0.3)

ax = axes[2]
ax.plot(sig['t_abs'], sig['v_pos'], color='#1f77b4', linewidth=0.7, label='voucher position')
if DELTA_HEDGE:
    ax.plot(sig['t_abs'], sig['vev_pos'], color='#d62728', linewidth=0.7, alpha=0.8, label='VEV hedge')
ax.axhline(0, color='black', linewidth=0.4)
ax.set_ylabel('position')
ax.grid(True, linewidth=0.3)
ax.legend(loc='upper right', fontsize=8)

ax = axes[3]
ax.plot(sig['t_abs'], sig['pnl'], color='#2ca02c', linewidth=0.9, label='PnL (MTM)')
ax.axhline(0, color='black', linewidth=0.4)
ax.set_ylabel('PnL')
ax.set_xlabel('Timestamp (3 days concatenated)')
ax.grid(True, linewidth=0.3)
ax.legend(loc='upper left', fontsize=8)

for axx in axes:
    for d in DAYS[1:]:
        axx.axvline(d * 1_000_000, color='black', linestyle='--', linewidth=0.5, alpha=0.4)

fig.tight_layout()
out = f'backtest_{VOUCHER}.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
print(f'saved {out}')
