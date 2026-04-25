import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ---- Config ----
DAYS = [0, 1, 2]
DATA_DIR = 'ROUND_3Data'
OUT_IMG = 'dashboard.png'

VOUCHERS = [
    ('VEV_4000', 4000), ('VEV_4500', 4500), ('VEV_5000', 5000),
    ('VEV_5100', 5100), ('VEV_5200', 5200), ('VEV_5300', 5300),
    ('VEV_5400', 5400), ('VEV_5500', 5500), ('VEV_6000', 6000),
    ('VEV_6500', 6500),
]
UNDERLYING = 'VELVETFRUIT_EXTRACT'


# ---- Data helpers (price = avg of highest-volume best bid / ask per tick) ----
def load_day(day):
    return pd.read_csv(f'{DATA_DIR}/prices_round_3_day_{day}.csv', sep=';')


def best_by_volume(df, side):
    rows = []
    for lvl in [1, 2, 3]:
        tmp = df[['timestamp', f'{side}_price_{lvl}', f'{side}_volume_{lvl}']].copy()
        tmp.columns = ['timestamp', 'price', 'volume']
        rows.append(tmp)
    stacked = pd.concat(rows).dropna()
    idx = stacked.groupby('timestamp')['volume'].idxmax()
    return stacked.loc[idx, ['timestamp', 'price']].sort_values('timestamp').reset_index(drop=True)


_cache = {}
def day_frame(day):
    if day not in _cache:
        _cache[day] = load_day(day)
    return _cache[day]


def price_series(product, day):
    df = day_frame(day)
    sub = df[df['product'] == product]
    if sub.empty:
        return pd.DataFrame(columns=['t_abs', 'price'])
    bb = best_by_volume(sub, 'bid').rename(columns={'price': 'bid'})
    ba = best_by_volume(sub, 'ask').rename(columns={'price': 'ask'})
    m = bb.merge(ba, on='timestamp', how='outer').sort_values('timestamp')
    m['price'] = (m['bid'] + m['ask']) / 2
    m['t_abs'] = m['timestamp'] + day * 1_000_000
    return m[['t_abs', 'price']]


def full_series(product):
    parts = [price_series(product, d) for d in DAYS]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts).sort_values('t_abs').reset_index(drop=True) if parts else pd.DataFrame()


# ---- Build figure ----
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)
ax_hg, ax_vev = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax_vch, ax_smile = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

day_bounds = [d * 1_000_000 for d in DAYS[1:]]


def mark_days(ax, label=False):
    for b in day_bounds:
        ax.axvline(b, color='black', linestyle='--', linewidth=0.9, alpha=0.55)
    if label:
        ymin, ymax = ax.get_ylim()
        for d in DAYS:
            ax.text(d * 1_000_000 + 500_000, ymax, f'day {d}',
                    ha='center', va='top', fontsize=9, color='dimgrey',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))


# Panel 1: HYDROGEL_PACK
hg = full_series('HYDROGEL_PACK')
ax_hg.plot(hg.t_abs, hg.price, color='#1f77b4', linewidth=0.7)
ax_hg.axhline(10000, color='red', linestyle='--', linewidth=0.6, alpha=0.7, label='10000')
ax_hg.set_title(f'HYDROGEL_PACK  (mean={hg.price.mean():.1f}, std={hg.price.std():.2f})')
ax_hg.set_ylabel('Price')
ax_hg.grid(True, linewidth=0.3)
ax_hg.legend(loc='upper right', fontsize=8)
mark_days(ax_hg, label=True)

# Panel 2: VELVETFRUIT_EXTRACT underlying
vev = full_series(UNDERLYING)
ax_vev.plot(vev.t_abs, vev.price, color='#ff7f0e', linewidth=0.7)
ret = np.diff(np.log(vev.price.values))
per_day_vol = ret.std() * np.sqrt(10000)
ax_vev.set_title(f'VELVETFRUIT_EXTRACT  (mean={vev.price.mean():.1f}, daily vol={per_day_vol:.3%})')
ax_vev.set_ylabel('Price')
ax_vev.grid(True, linewidth=0.3)
mark_days(ax_vev, label=True)

# Panel 3: all vouchers, log y
cmap = cm.get_cmap('viridis', len(VOUCHERS))
for i, (name, strike) in enumerate(VOUCHERS):
    s = full_series(name)
    if s.empty:
        continue
    ax_vch.plot(s.t_abs, s.price, color=cmap(i), linewidth=0.7, label=f'{name} (K={strike})')
ax_vch.set_yscale('log')
ax_vch.set_title('VEV vouchers (call options, log y)')
ax_vch.set_ylabel('Price (log)')
ax_vch.set_xlabel('Timestamp (days 0,1,2 concatenated)')
ax_vch.grid(True, linewidth=0.3, which='both')
ax_vch.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
mark_days(ax_vch, label=True)

# Panel 4: option smile — price vs strike at mid-day snapshot of each day
snapshots = []
colors = ['#2ca02c', '#d62728', '#9467bd']
for di, day in enumerate(DAYS):
    t_snap = day * 1_000_000 + 500_000
    s_vev = vev[vev.t_abs == t_snap]
    if s_vev.empty:
        continue
    s_underlying = float(s_vev.price.iloc[0])
    strikes, prices = [], []
    for name, strike in VOUCHERS:
        s = full_series(name)
        row = s[s.t_abs == t_snap]
        if row.empty:
            continue
        strikes.append(strike)
        prices.append(float(row.price.iloc[0]))
    ax_smile.plot(strikes, prices, 'o-', color=colors[di], linewidth=1.2, markersize=5,
                  label=f'day {day}  (S={s_underlying:.1f})')
    snapshots.append((day, s_underlying))

# Intrinsic value line for the first snapshot's underlying (visual anchor)
if snapshots:
    S_ref = snapshots[0][1]
    k_line = np.linspace(min(k for _, k in VOUCHERS), max(k for _, k in VOUCHERS), 200)
    intr = np.maximum(S_ref - k_line, 0)
    ax_smile.plot(k_line, intr, color='grey', linestyle='--', linewidth=0.8,
                  label=f'intrinsic @ S={S_ref:.0f}')
ax_smile.set_title('Option smile  —  voucher price vs strike  (snapshot t=ts+500k each day)')
ax_smile.set_xlabel('Strike K')
ax_smile.set_ylabel('Voucher price')
ax_smile.grid(True, linewidth=0.3)
ax_smile.legend(fontsize=8, loc='upper right')

fig.suptitle('Round 3 "Gloves Off" — market overview', fontsize=13, y=0.995)
fig.savefig(OUT_IMG, dpi=140, bbox_inches='tight')
print(f'saved {OUT_IMG}')


# ---- Second figure: smile using time-averaged prices over all 3 days ----
avg_S = vev.price.mean()
avg_prices, strikes = [], []
for name, strike in VOUCHERS:
    s = full_series(name)
    if s.empty:
        continue
    avg_prices.append(s.price.mean())
    strikes.append(strike)

avg_prices = np.array(avg_prices)
strikes = np.array(strikes)
intrinsic = np.maximum(avg_S - strikes, 0)
extrinsic = avg_prices - intrinsic

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(strikes, avg_prices, 'o-', color='#d62728', linewidth=1.5, markersize=7,
         label='avg voucher price (3-day mean)')
k_line = np.linspace(strikes.min(), strikes.max(), 200)
ax2.plot(k_line, np.maximum(avg_S - k_line, 0), color='grey', linestyle='--', linewidth=1.0,
         label=f'intrinsic @ avg S = {avg_S:.1f}')

for k, p, e in zip(strikes, avg_prices, extrinsic):
    ax2.annotate(f'{p:.1f}\n(ext {e:+.1f})', xy=(k, p),
                 xytext=(6, 6), textcoords='offset points', fontsize=7, color='#444')

ax2.set_title(f'Time-averaged option smile  —  avg price vs strike  (avg S = {avg_S:.1f})')
ax2.set_xlabel('Strike K')
ax2.set_ylabel('Average voucher price (3-day mean)')
ax2.grid(True, linewidth=0.3)
ax2.legend(loc='upper right', fontsize=9)

fig2.tight_layout()
fig2.savefig('avgSmile.png', dpi=140, bbox_inches='tight')
print('saved avgSmile.png')

print('\nTime-averaged voucher prices:')
print(f'  avg underlying S = {avg_S:.2f}')
for k, p, e in zip(strikes, avg_prices, extrinsic):
    print(f'  K={k:5d}  avg price={p:8.2f}  intrinsic={max(avg_S-k,0):7.2f}  extrinsic={e:+7.2f}')
print(f'\nUnderlying VEV realized vol (per day, sqrt(10000 ticks)): {per_day_vol:.4%}')
for name, strike in VOUCHERS:
    s = full_series(name)
    if not s.empty:
        print(f'  {name:10s} K={strike:5d}   price range [{s.price.min():7.2f}, {s.price.max():7.2f}]   mean={s.price.mean():7.2f}')
