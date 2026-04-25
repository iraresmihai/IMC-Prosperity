import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Config ----
DAYS = [0, 1, 2]
DATA_DIR = 'ROUND_3Data'
PRODUCTS = [
    'HYDROGEL_PACK', 'VELVETFRUIT_EXTRACT',
    'VEV_4000', 'VEV_4500', 'VEV_5000', 'VEV_5100', 'VEV_5200',
    'VEV_5300', 'VEV_5400', 'VEV_5500', 'VEV_6000', 'VEV_6500',
]
TICKS_PER_DAY = 10000  # timestamps 0..999900 step 100


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


def day_price_matrix(day):
    """Return DataFrame: index=timestamp, columns=products, values=mid price."""
    df = pd.read_csv(f'{DATA_DIR}/prices_round_3_day_{day}.csv', sep=';')
    cols = {}
    for p in PRODUCTS:
        sub = df[df['product'] == p]
        if sub.empty:
            continue
        bb = best_by_volume(sub, 'bid').rename(columns={'price': 'bid'})
        ba = best_by_volume(sub, 'ask').rename(columns={'price': 'ask'})
        m = bb.merge(ba, on='timestamp', how='inner')
        m['mid'] = (m['bid'] + m['ask']) / 2
        cols[p] = m.set_index('timestamp')['mid']
    return pd.DataFrame(cols).sort_index()


# ---- Load all days ----
day_mats = {d: day_price_matrix(d) for d in DAYS}


# ---- Volatility table (per-day, per-product; %, scaled to per-day via sqrt(ticks)) ----
vol_rows = {}
for d, mat in day_mats.items():
    col = {}
    for p in mat.columns:
        prices = mat[p].dropna().values
        if len(prices) < 2 or np.all(prices == prices[0]):
            col[p] = 0.0
        else:
            log_ret = np.diff(np.log(prices))
            col[p] = log_ret.std() * np.sqrt(TICKS_PER_DAY) * 100  # per-day vol in %
    vol_rows[f'day_{d}'] = col

vol_df = pd.DataFrame(vol_rows).reindex(PRODUCTS)
vol_df['mean'] = vol_df.mean(axis=1)

print('=' * 72)
print('Realized volatility per day  (per-day %, tick-log-return std * sqrt(10000))')
print('=' * 72)
print(vol_df.round(3).to_string())


# ---- Correlation matrix (tick-to-tick log returns, concatenated across 3 days) ----
returns_per_day = []
for d, mat in day_mats.items():
    logp = np.log(mat)
    ret = logp.diff().dropna(how='all')
    returns_per_day.append(ret)
returns_all = pd.concat(returns_per_day)

corr = returns_all.corr().reindex(index=PRODUCTS, columns=PRODUCTS)

print('\n' + '=' * 72)
print('Correlation of tick-to-tick log returns  (all 3 days concatenated)')
print('=' * 72)
# short labels so the table fits
short = {p: p.replace('VELVETFRUIT_EXTRACT', 'VEV').replace('HYDROGEL_PACK', 'HG') for p in PRODUCTS}
corr_short = corr.rename(index=short, columns=short)
with pd.option_context('display.width', 160, 'display.max_columns', None):
    print(corr_short.round(2).to_string())


# ---- Save CSVs ----
vol_df.to_csv('volatility_table.csv')
corr.to_csv('correlation_table.csv')


# ---- Heatmap PNG for correlation ----
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(corr_short.values.astype(float), cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_short)))
ax.set_yticks(range(len(corr_short)))
ax.set_xticklabels(corr_short.columns, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(corr_short.index, fontsize=9)
for i in range(len(corr_short)):
    for j in range(len(corr_short)):
        v = corr_short.iloc[i, j]
        if pd.isna(v):
            txt = 'n/a'
            c = 'grey'
        else:
            txt = f'{v:.2f}'
            c = 'white' if abs(v) > 0.6 else 'black'
        ax.text(j, i, txt, ha='center', va='center', fontsize=7, color=c)
plt.colorbar(im, ax=ax, label='Pearson correlation')
ax.set_title('Log-return correlation matrix  (3 days concatenated)')
fig.tight_layout()
fig.savefig('correlationHeatmap.png', dpi=140, bbox_inches='tight')
print('\nsaved: volatility_table.csv, correlation_table.csv, correlationHeatmap.png')
