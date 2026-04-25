"""
"Smart" backtest combining structural bias + intra-round mean reversion on
ONE VEV voucher, using a FIXED flat IV as the fair-value benchmark.

Logic per tick:
  1) S = mid(VEV)
  2) C_fair  = bs_call(S, K, T, SIGMA_FLAT)
  3) residual_$ = C_market - C_fair
  4) Linear sizing:
        target_pos = clip( -K_SIZE * residual_$,  -POSITION_LIMIT, +POSITION_LIMIT )
     -> persistently long when residual is negative (cheap), short when positive,
        AND scales harder when the deviation is bigger.
  5) Optional delta hedge (delta computed at SIGMA_FLAT).
  6) MTM pnl tracked tick-by-tick; final pnl ALSO computed by liquidating at
     bs_call(SIGMA_FLAT) — that's how IMC settles open positions.
  7) Optional MIN_TRADE_QTY (don't trade tiny rebalances) to control churn cost.

Configurable up top:
  VOUCHER      which strike to trade
  SIGMA_FLAT   the assumed flat IV (per-day) used both as fair value AND for
               liquidation. Try variations to see sensitivity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt, erf, exp

# =================== USER CONFIG ===================
VOUCHER          = 'VEV_5300'   # which voucher to trade
SIGMA_FLAT       = 0.01262      # per-day sigma; ~20% annualized
K_SIZE           = 150          # position units per $1 of mispricing
POSITION_LIMIT   = 300          # voucher position cap
VEV_LIMIT        = 200          # underlying position cap
MIN_TRADE_QTY    = 5            # ignore rebalances smaller than this
RESIDUAL_DEADZONE = 0.50        # only retarget if |resid - last_traded_resid| > this ($)
DELTA_HEDGE      = True
CROSS_SPREAD     = True         # realistic execution: pay ask, receive bid
DAYS             = [0, 1, 2]
SWEEP_DEADZONES  = [0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50]  # for end-of-run sweep
# ===================================================

DATA_DIR    = 'ROUND_3Data'
UNDERLYING  = 'VELVETFRUIT_EXTRACT'
VOUCHERS_ALL = [
    ('VEV_4000', 4000), ('VEV_4500', 4500), ('VEV_5000', 5000),
    ('VEV_5100', 5100), ('VEV_5200', 5200), ('VEV_5300', 5300),
    ('VEV_5400', 5400), ('VEV_5500', 5500), ('VEV_6000', 6000),
    ('VEV_6500', 6500),
]
STRIKE_MAP = dict(VOUCHERS_ALL)
TTE_START  = {0: 8.0, 1: 7.0, 2: 6.0}
K_TRADE    = STRIKE_MAP[VOUCHER]


# ---- Price helpers ----
def best_by_vol(df, side):
    rows = []
    for lvl in [1, 2, 3]:
        tmp = df[['timestamp', f'{side}_price_{lvl}', f'{side}_volume_{lvl}']].copy()
        tmp.columns = ['timestamp', 'price', 'volume']
        rows.append(tmp)
    st = pd.concat(rows).dropna()
    idx = st.groupby('timestamp')['volume'].idxmax()
    return (st.loc[idx, ['timestamp', 'price']]
            .sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True))


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


# ---- Load data ----
print(f'Loading days {DAYS}...')
day_data = {}
for d in DAYS:
    raw = pd.read_csv(f'{DATA_DIR}/prices_round_3_day_{d}.csv', sep=';')
    day_data[d] = {
        VOUCHER: prod_frame(raw, VOUCHER),
        UNDERLYING: prod_frame(raw, UNDERLYING),
    }


# ---- Build per-tick signal ----
print(f'Computing per-tick fair value at SIGMA_FLAT = {SIGMA_FLAT*100:.4f}%/day...')
rows = []
for d in DAYS:
    vev_df = day_data[d][UNDERLYING]
    voucher_df = day_data[d][VOUCHER]
    common = vev_df.index.intersection(voucher_df.index)

    for ts in common:
        S = float(vev_df.loc[ts, 'mid'])
        T = TTE_START[d] - ts / 1_000_000
        if T <= 0:
            continue
        C_fair = bs_call(S, K_TRADE, T, SIGMA_FLAT)
        delta  = bs_delta(S, K_TRADE, T, SIGMA_FLAT)
        rows.append({
            't_abs': ts + d * 1_000_000, 'day': d, 'ts': ts,
            'S': S, 'T': T,
            'v_mid': float(voucher_df.loc[ts, 'mid']),
            'v_bid': float(voucher_df.loc[ts, 'bid']),
            'v_ask': float(voucher_df.loc[ts, 'ask']),
            'vev_mid': S,
            'vev_bid': float(vev_df.loc[ts, 'bid']),
            'vev_ask': float(vev_df.loc[ts, 'ask']),
            'C_fair': C_fair, 'delta': delta,
            'resid_$': float(voucher_df.loc[ts, 'mid']) - C_fair,
        })

sig = pd.DataFrame(rows).sort_values('t_abs').reset_index(drop=True)
print(f'  ticks: {len(sig)}')


resid   = sig['resid_$'].to_numpy()
deltas  = sig['delta'].to_numpy()
v_mid   = sig['v_mid'].to_numpy()
v_bid   = sig['v_bid'].to_numpy()
v_ask   = sig['v_ask'].to_numpy()
vev_mid = sig['vev_mid'].to_numpy()
vev_bid = sig['vev_bid'].to_numpy()
vev_ask = sig['vev_ask'].to_numpy()
C_fair  = sig['C_fair'].to_numpy()


def run_strategy(deadzone, k_size=K_SIZE, cross_spread=CROSS_SPREAD,
                 delta_hedge=DELTA_HEDGE, min_trade=MIN_TRADE_QTY):
    """Run one strategy instance; return arrays + counters."""
    n = len(sig)
    v_pos = vev_pos = 0
    cash = 0.0
    n_v = n_vev = 0
    last_traded_resid = None       # residual at last voucher trade

    v_pos_arr   = np.zeros(n, dtype=int)
    vev_pos_arr = np.zeros(n, dtype=int)
    mtm_arr     = np.zeros(n)
    fair_arr    = np.zeros(n)

    for i in range(n):
        # only re-target if residual has moved enough since last trade
        retarget = (last_traded_resid is None
                    or abs(resid[i] - last_traded_resid) > deadzone)

        if retarget:
            target = int(round(-k_size * resid[i]))
            target = max(-POSITION_LIMIT, min(POSITION_LIMIT, target))
        else:
            target = v_pos

        qty = target - v_pos
        if abs(qty) >= min_trade:
            px = (v_ask[i] if qty > 0 else v_bid[i]) if cross_spread else v_mid[i]
            cash -= qty * px
            v_pos = target
            n_v += 1
            last_traded_resid = resid[i]

        if delta_hedge:
            tgt_vev = int(round(-v_pos * deltas[i]))
            tgt_vev = max(-VEV_LIMIT, min(VEV_LIMIT, tgt_vev))
            dv = tgt_vev - vev_pos
            if abs(dv) >= min_trade:
                pvev = (vev_ask[i] if dv > 0 else vev_bid[i]) if cross_spread else vev_mid[i]
                cash -= dv * pvev
                vev_pos = tgt_vev
                n_vev += 1

        mtm_arr[i]   = cash + v_pos * v_mid[i]  + vev_pos * vev_mid[i]
        fair_arr[i]  = cash + v_pos * C_fair[i] + vev_pos * vev_mid[i]
        v_pos_arr[i] = v_pos
        vev_pos_arr[i] = vev_pos

    return v_pos_arr, vev_pos_arr, mtm_arr, fair_arr, n_v, n_vev


print(f'Running smart strategy (deadzone = ${RESIDUAL_DEADZONE:.2f})...')
(v_pos_arr, vev_pos_arr, mtm_arr, fair_arr,
 n_v_trades, n_vev_trades) = run_strategy(RESIDUAL_DEADZONE)

sig['v_pos']    = v_pos_arr
sig['vev_pos']  = vev_pos_arr
sig['pnl_mtm']  = mtm_arr
sig['pnl_fair'] = fair_arr


# ---- Summary ----
peak = np.maximum.accumulate(mtm_arr)
dd = mtm_arr - peak

print('\n' + '=' * 56)
print('SMART BACKTEST RESULTS')
print('=' * 56)
print(f'Voucher:               {VOUCHER}  (K = {K_TRADE})')
print(f'SIGMA_FLAT:            {SIGMA_FLAT:.6f}/day  ({SIGMA_FLAT*sqrt(252)*100:.2f}% annual)')
print(f'K_SIZE:                {K_SIZE}  (units per $1 of mispricing)')
print(f'POSITION_LIMIT:        ±{POSITION_LIMIT}')
print(f'Delta hedge:           {DELTA_HEDGE}')
print(f'Cross spread:          {CROSS_SPREAD}')
print(f'Days:                  {DAYS}')
print(f'Voucher trades:        {n_v_trades}')
print(f'VEV hedge trades:      {n_vev_trades}')
print('-' * 56)
print(f'PnL  (MTM, market):     {mtm_arr[-1]:+.2f}')
print(f'PnL  (fair, IMC settle):{fair_arr[-1]:+.2f}    <-- this is what gets paid')
print(f'Max MTM drawdown:       {dd.min():.2f}')
print('-' * 56)
for d in DAYS:
    sub = sig[sig['day'] == d]
    if sub.empty: continue
    prev_end = sig[sig['day'] < d]['pnl_mtm'].iloc[-1] if d > DAYS[0] else 0.0
    prev_end_f = sig[sig['day'] < d]['pnl_fair'].iloc[-1] if d > DAYS[0] else 0.0
    print(f'  day {d}:    MTM {sub["pnl_mtm"].iloc[-1] - prev_end:+9.2f}   '
          f'fair {sub["pnl_fair"].iloc[-1] - prev_end_f:+9.2f}')
print('=' * 56)


# ---- Plot ----
fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True,
                         gridspec_kw={'height_ratios': [1, 1, 1, 1, 1.3]})

ax = axes[0]
ax.plot(sig['t_abs'], sig['v_mid'],  color='#1f77b4', linewidth=0.7, label=f'{VOUCHER} mid')
ax.plot(sig['t_abs'], sig['C_fair'], color='#d62728', linewidth=0.7, alpha=0.8,
        label=f'BS fair @ σ={SIGMA_FLAT*100:.3f}%/day')
ax.set_ylabel('price')
ax.set_title(f'Smart backtest — {VOUCHER}  '
             f'|  σ_flat={SIGMA_FLAT*100:.4f}%/day  '
             f'|  k={K_SIZE}  |  deadzone=${RESIDUAL_DEADZONE:.2f}  '
             f'|  hedge={DELTA_HEDGE}  |  cross_spread={CROSS_SPREAD}')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, linewidth=0.3)

ax = axes[1]
ax.plot(sig['t_abs'], sig['resid_$'], color='#9467bd', linewidth=0.6)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('residual ($)')
ax.set_title('C_market - C_fair')
ax.grid(True, linewidth=0.3)

ax = axes[2]
ax.plot(sig['t_abs'], sig['v_pos'], color='#1f77b4', linewidth=0.7, label='voucher pos')
if DELTA_HEDGE:
    ax.plot(sig['t_abs'], sig['vev_pos'], color='#d62728', linewidth=0.7, alpha=0.8, label='VEV hedge')
ax.axhline(0, color='black', linewidth=0.4)
ax.set_ylabel('position')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, linewidth=0.3)

ax = axes[3]
ax.plot(sig['t_abs'], sig['pnl_mtm'],  color='#2ca02c', linewidth=0.9, label='PnL (MTM, market)')
ax.plot(sig['t_abs'], sig['pnl_fair'], color='#ff7f0e', linewidth=0.9, label='PnL (fair, settlement)')
ax.axhline(0, color='black', linewidth=0.4)
ax.set_ylabel('PnL')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, linewidth=0.3)

ax = axes[4]
ax.fill_between(sig['t_abs'], dd, 0, color='#d62728', alpha=0.5)
ax.set_ylabel('MTM drawdown')
ax.set_xlabel('Timestamp (3 days concatenated)')
ax.grid(True, linewidth=0.3)

for axx in axes:
    for d in DAYS[1:]:
        axx.axvline(d * 1_000_000, color='black', linestyle='--', linewidth=0.5, alpha=0.4)

fig.tight_layout()
out = f'smartBacktest_{VOUCHER}_sigma{SIGMA_FLAT*1e5:.0f}.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
print(f'saved {out}')


# ---- Deadzone sweep (BOTH spread modes side-by-side) ----
def sweep(cross):
    print('\n' + '=' * 72)
    print(f'DEADZONE SWEEP   voucher={VOUCHER}  sigma={SIGMA_FLAT:.5f}  '
          f'cross_spread={cross}')
    print('=' * 72)
    print(f'{"deadzone":>10} | {"v_trades":>9} {"vev_trades":>11} | '
          f'{"PnL_MTM":>11} {"PnL_fair":>11} {"max_DD":>10}')
    print('-' * 72)
    rows = []
    for dz in SWEEP_DEADZONES:
        _, _, mtm_s, fair_s, nv, nh = run_strategy(dz, cross_spread=cross)
        peak_s = np.maximum.accumulate(mtm_s)
        dd_s = (mtm_s - peak_s).min()
        print(f'{dz:>10.2f} | {nv:>9d} {nh:>11d} | '
              f'{mtm_s[-1]:>+11.0f} {fair_s[-1]:>+11.0f} {dd_s:>10.0f}')
        rows.append({'deadzone': dz, 'v_trades': nv, 'vev_trades': nh,
                     'pnl_mtm': mtm_s[-1], 'pnl_fair': fair_s[-1], 'max_dd': dd_s})
    return pd.DataFrame(rows)


sweep_mid   = sweep(False)
sweep_cross = sweep(True)

best_mid = sweep_mid.iloc[sweep_mid['pnl_fair'].idxmax()]
best_cross = sweep_cross.iloc[sweep_cross['pnl_fair'].idxmax()]
print('\nBest deadzone @ mid:    {:.2f}  -> fair PnL {:+.0f}'.format(
    best_mid['deadzone'], best_mid['pnl_fair']))
print('Best deadzone @ cross:  {:.2f}  -> fair PnL {:+.0f}'.format(
    best_cross['deadzone'], best_cross['pnl_fair']))

# Sweep plot
figs, axs = plt.subplots(1, 2, figsize=(13, 4.5))
axs[0].plot(sweep_mid['deadzone'],   sweep_mid['pnl_fair'],   'o-',
            color='#2ca02c', label='mid (cross_spread=False)')
axs[0].plot(sweep_cross['deadzone'], sweep_cross['pnl_fair'], 's-',
            color='#d62728', label='realistic (cross_spread=True)')
axs[0].axhline(0, color='black', linewidth=0.4)
axs[0].set_xlabel('residual deadzone ($)')
axs[0].set_ylabel('Final PnL (fair-marked)')
axs[0].set_title(f'PnL vs deadzone — {VOUCHER}, sigma={SIGMA_FLAT*100:.3f}%/day')
axs[0].grid(True, linewidth=0.3); axs[0].legend(fontsize=9)

axs[1].plot(sweep_mid['deadzone'],   sweep_mid['v_trades'],   'o-',
            color='#2ca02c', label='voucher trades (mid)')
axs[1].plot(sweep_cross['deadzone'], sweep_cross['v_trades'], 's-',
            color='#d62728', label='voucher trades (cross)')
axs[1].set_xlabel('residual deadzone ($)')
axs[1].set_ylabel('# voucher trades')
axs[1].set_title('Trade count vs deadzone')
axs[1].grid(True, linewidth=0.3); axs[1].legend(fontsize=9)

figs.tight_layout()
sweep_out = f'smartBacktest_sweep_{VOUCHER}.png'
figs.savefig(sweep_out, dpi=140, bbox_inches='tight')
print(f'saved {sweep_out}')
