import pandas as pd
import matplotlib.pyplot as plt

# ---- Configuration: edit these two lists to pick what to plot ----
PRODUCTS = [
    'HYDROGEL_PACK',
    'VELVETFRUIT_EXTRACT',
    'VEV_4000',
    'VEV_4500',
    'VEV_5000',
    'VEV_5100',
    'VEV_5200',
    'VEV_5300',
    'VEV_5400',
    'VEV_5500',
    'VEV_6000',
    'VEV_6500',
]

DAYS = [0, 1, 2]

CONCAT_DAYS = False        # True -> chain days on one timeline; False -> overlay on same ts
DATA_DIR = 'ROUND_3Data'
# ------------------------------------------------------------------


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


def build_price(product, day):
    df = load_day(day)
    sub = df[df['product'] == product]
    if sub.empty:
        return pd.DataFrame(columns=['timestamp', 'price'])

    best_bids = best_by_volume(sub, 'bid').rename(columns={'price': 'bid'})
    best_asks = best_by_volume(sub, 'ask').rename(columns={'price': 'ask'})
    merged = best_bids.merge(best_asks, on='timestamp', how='outer').sort_values('timestamp')
    merged['price'] = (merged['bid'] + merged['ask']) / 2
    if CONCAT_DAYS:
        merged['timestamp'] = merged['timestamp'] + day * 1_000_000
    return merged[['timestamp', 'price']]


PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#393b79', '#637939',
]
LINESTYLES = {0: '-', 1: '--', 2: ':'}

fig, ax = plt.subplots(figsize=(13, 7))

for i, product in enumerate(PRODUCTS):
    color = PALETTE[i % len(PALETTE)]
    for day in DAYS:
        s = build_price(product, day)
        if s.empty:
            continue
        label_day = '' if len(DAYS) == 1 else f' d{day}'
        ax.plot(s['timestamp'], s['price'],
                color=color, linestyle=LINESTYLES.get(day, '-'),
                linewidth=1.2, label=f'{product}{label_day}')

ax.set_title(f'Round 3 prices  |  days {DAYS}')
ax.set_xlabel('Timestamp' + (' (concatenated across days)' if CONCAT_DAYS else ''))
ax.set_ylabel('Price')
ax.grid(True, color='lightgrey', linewidth=0.5)
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig('priceVisualizer.png', dpi=140, bbox_inches='tight')
plt.show()
