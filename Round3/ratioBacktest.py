"""
Pair / ratio stat-arb on two VEV vouchers — no Black-Scholes.

Logic per tick:
  ratio = mid_A / mid_B
  rolling mean, std over WINDOW ticks
  z = (ratio - mean) / std

  z > +ENTRY_Z   ->  ratio too high  ->  SHORT A, LONG B
  z < -ENTRY_Z   ->  ratio too low   ->  LONG A,  SHORT B
  |z| < EXIT_Z   ->  flat
  |z| > EMERGENCY_Z -> force flat (stop-loss / regime break)
  else           ->  hold current position

Sizing: equal-dollar notional, capped by per-leg position limit.
  target $ exposure E = min(POSITION_LIMIT * mid_A, POSITION_LIMIT * mid_B)
  qty_A =  sign * round(E / mid_A)    qty_B = -sign * round(E / mid_B)
  (both clipped to ±POSITION_LIMIT for safety)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =================== USER CONFIG ===================
PAIR             = ('VEV_5400', 'VEV_5300')   # (A, B)  -> trade ratio A / B
WINDOW           = 1000
ENTRY_Z          = 2.0
EXIT_Z           = 0.5
EMERGENCY_Z      = 4.0
POSITION_LIMIT   = 300                  # cap per voucher
CROSS_SPREAD     = True                 # True = pay ask / receive bid
DAYS             = [0, 1, 2]
# ===================================================

DATA_DIR = 'ROUND_3Data'
A, B = PAIR


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


# ---- Load ----
print(f'Loading {A} and {B} for days {DAYS}...')
frames = []
for d in DAYS:
    raw = pd.read_csv(f'{DATA_DIR}/prices_round_3_day_{d}.csv', sep=';')
    fa = prod_frame(raw, A)
    fb = prod_frame(raw, B)
    common = fa.index.intersection(fb.index)
    df = pd.DataFrame({
        'a_mid': fa.loc[common, 'mid'].values,
        'a_bid': fa.loc[common, 'bid'].values,
        'a_ask': fa.loc[common, 'ask'].values,
        'b_mid': fb.loc[common, 'mid'].values,
        'b_bid': fb.loc[common, 'bid'].values,
        'b_ask': fb.loc[common, 'ask'].values,
    }, index=common)
    df['day'] = d
    df['t_abs'] = df.index + d * 1_000_000
    frames.append(df.reset_index().rename(columns={'index': 'ts'}))

sig = pd.concat(frames, ignore_index=True).sort_values('t_abs').reset_index(drop=True)
print(f'  ticks: {len(sig)}')


# ---- Signal ----
sig['ratio']   = sig['a_mid'] / sig['b_mid']
sig['mean']    = sig['ratio'].rolling(WINDOW, min_periods=WINDOW).mean()
sig['std']     = sig['ratio'].rolling(WINDOW, min_periods=WINDOW).std()
sig['z']       = (sig['ratio'] - sig['mean']) / sig['std']

a_mid = sig['a_mid'].to_numpy()
a_bid = sig['a_bid'].to_numpy()
a_ask = sig['a_ask'].to_numpy()
b_mid = sig['b_mid'].to_numpy()
b_bid = sig['b_bid'].to_numpy()
b_ask = sig['b_ask'].to_numpy()
z_arr = sig['z'].to_numpy()
n = len(sig)


def target_qty(sign_a, pa, pb):
    """Equal-$ sizing capped by POSITION_LIMIT per leg.
       sign_a = +1 -> long A short B;  -1 -> short A long B."""
    if sign_a == 0:
        return 0, 0
    E = min(POSITION_LIMIT * pa, POSITION_LIMIT * pb)
    qa =  sign_a * int(round(E / pa))
    qb = -sign_a * int(round(E / pb))
    qa = max(-POSITION_LIMIT, min(POSITION_LIMIT, qa))
    qb = max(-POSITION_LIMIT, min(POSITION_LIMIT, qb))
    return qa, qb


def run(entry_z, exit_z, emergency_z, cross_spread):
    pos_a = pos_b = 0
    cash = 0.0
    trades_a = trades_b = 0

    pa_arr = np.zeros(n, dtype=int)
    pb_arr = np.zeros(n, dtype=int)
    pnl_arr = np.zeros(n)
    state_arr = np.zeros(n, dtype=int)   # -1 short A long B, +1 long A short B, 0 flat

    state = 0
    for i in range(n):
        z = z_arr[i]
        if np.isnan(z):
            pa_arr[i] = pos_a; pb_arr[i] = pos_b
            pnl_arr[i] = cash + pos_a * a_mid[i] + pos_b * b_mid[i]
            state_arr[i] = state
            continue

        new_state = state
        if abs(z) > emergency_z:
            new_state = 0
        elif state == 0:
            if z > entry_z:
                new_state = -1   # short A, long B
            elif z < -entry_z:
                new_state = +1   # long A, short B
        else:
            if abs(z) < exit_z:
                new_state = 0

        if new_state != state:
            if new_state == 0:
                tgt_a, tgt_b = 0, 0
            else:
                tgt_a, tgt_b = target_qty(new_state, a_mid[i], b_mid[i])

            da = tgt_a - pos_a
            db = tgt_b - pos_b
            if da != 0:
                pa = (a_ask[i] if da > 0 else a_bid[i]) if cross_spread else a_mid[i]
                cash -= da * pa
                pos_a = tgt_a
                trades_a += 1
            if db != 0:
                pb = (b_ask[i] if db > 0 else b_bid[i]) if cross_spread else b_mid[i]
                cash -= db * pb
                pos_b = tgt_b
                trades_b += 1
            state = new_state

        pa_arr[i] = pos_a
        pb_arr[i] = pos_b
        pnl_arr[i] = cash + pos_a * a_mid[i] + pos_b * b_mid[i]
        state_arr[i] = state

    return pa_arr, pb_arr, pnl_arr, state_arr, trades_a, trades_b


print(f'Running ratio stat-arb on {A}/{B}  '
      f'(entry={ENTRY_Z}, exit={EXIT_Z}, emergency={EMERGENCY_Z}, '
      f'cross_spread={CROSS_SPREAD})...')
pa_arr, pb_arr, pnl_arr, state_arr, ta, tb = run(
    ENTRY_Z, EXIT_Z, EMERGENCY_Z, CROSS_SPREAD)

sig['pos_a']  = pa_arr
sig['pos_b']  = pb_arr
sig['pnl']    = pnl_arr
sig['state']  = state_arr

peak = np.maximum.accumulate(pnl_arr)
dd = pnl_arr - peak

print('\n' + '=' * 56)
print(f'RATIO BACKTEST  —  {A} / {B}')
print('=' * 56)
print(f'Window:          {WINDOW}')
print(f'Entry / Exit Z:  ±{ENTRY_Z} / ±{EXIT_Z}')
print(f'Emergency Z:     ±{EMERGENCY_Z}')
print(f'Position limit:  ±{POSITION_LIMIT} per voucher')
print(f'Cross spread:    {CROSS_SPREAD}')
print(f'Trades A / B:    {ta} / {tb}')
print(f'Final PnL:       {pnl_arr[-1]:+.2f}')
print(f'Max drawdown:    {dd.min():+.2f}')
for d in DAYS:
    sub = sig[sig['day'] == d]
    if sub.empty: continue
    prev = sig[sig['day'] < d]['pnl'].iloc[-1] if d > DAYS[0] else 0.0
    print(f'  day {d}:        {sub["pnl"].iloc[-1] - prev:+9.2f}')
print('=' * 56)


# ---- Plot ----
fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True,
                         gridspec_kw={'height_ratios': [1, 1, 1, 1, 1.1]})

ax = axes[0]
ax.plot(sig['t_abs'], sig['a_mid'], linewidth=0.7, label=A, color='#1f77b4')
ax.plot(sig['t_abs'], sig['b_mid'], linewidth=0.7, label=B, color='#d62728')
ax.set_ylabel('price'); ax.legend(fontsize=8); ax.grid(True, linewidth=0.3)
ax.set_title(f'Ratio stat-arb  {A}/{B}  '
             f'|  win={WINDOW}  entry=±{ENTRY_Z}  exit=±{EXIT_Z}  '
             f'emrg=±{EMERGENCY_Z}  |  cross_spread={CROSS_SPREAD}')

ax = axes[1]
ax.plot(sig['t_abs'], sig['ratio'], linewidth=0.6, color='#9467bd', label='ratio')
ax.plot(sig['t_abs'], sig['mean'],  linewidth=0.6, color='black', alpha=0.7, label='roll mean')
ax.set_ylabel('A / B'); ax.legend(fontsize=8); ax.grid(True, linewidth=0.3)

ax = axes[2]
ax.plot(sig['t_abs'], sig['z'], linewidth=0.5, color='#2ca02c')
for lvl, c, ls in [(ENTRY_Z, '#d62728', '--'), (-ENTRY_Z, '#d62728', '--'),
                   (EXIT_Z, 'black', ':'), (-EXIT_Z, 'black', ':'),
                   (EMERGENCY_Z, '#8b0000', '-'), (-EMERGENCY_Z, '#8b0000', '-')]:
    ax.axhline(lvl, color=c, linestyle=ls, linewidth=0.6, alpha=0.7)
ax.set_ylabel('z-score'); ax.grid(True, linewidth=0.3)

ax = axes[3]
ax.plot(sig['t_abs'], sig['pos_a'], linewidth=0.7, color='#1f77b4', label=f'pos {A}')
ax.plot(sig['t_abs'], sig['pos_b'], linewidth=0.7, color='#d62728', label=f'pos {B}')
ax.axhline(0, color='black', linewidth=0.4)
ax.set_ylabel('position'); ax.legend(fontsize=8); ax.grid(True, linewidth=0.3)

ax = axes[4]
ax.plot(sig['t_abs'], sig['pnl'], linewidth=0.9, color='#2ca02c', label='PnL')
ax.fill_between(sig['t_abs'], dd, 0, color='#d62728', alpha=0.3, label='drawdown')
ax.axhline(0, color='black', linewidth=0.4)
ax.set_ylabel('PnL'); ax.set_xlabel('Timestamp (3 days concatenated)')
ax.legend(fontsize=8); ax.grid(True, linewidth=0.3)

for axx in axes:
    for d in DAYS[1:]:
        axx.axvline(d * 1_000_000, color='black', linestyle='--',
                    linewidth=0.5, alpha=0.4)

fig.tight_layout()
out = f'ratioBacktest_{A}_{B}.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
print(f'saved {out}')
