import pandas as pd
import plotly.graph_objects as go
import json

volume_threshold = 10
product = 'INTARIAN_PEPPER_ROOT'
has_trading_history = True

# Load the CSV
df = pd.read_csv('tradingData.csv', sep=';')

# Filter one product
fig = go.Figure()

data = df[df['product'] == product].copy()

bid_rows = []
ask_rows = []
for lvl in [1, 2, 3]:
    temp = data[['timestamp', f'bid_price_{lvl}', f'bid_volume_{lvl}']].copy()
    temp.columns = ['timestamp', 'price', 'volume']
    bid_rows.append(temp)
    temp = data[['timestamp', f'ask_price_{lvl}', f'ask_volume_{lvl}']].copy()
    temp.columns = ['timestamp', 'price', 'volume']
    ask_rows.append(temp)

bids = pd.concat(bid_rows).dropna().reset_index(drop=True)
asks = pd.concat(ask_rows).dropna().reset_index(drop=True)

min_ask = asks.volume.min()
max_ask = asks.volume.max()
max_bid = bids.volume.max()
min_bid = bids.volume.min()

def scale_vol(info, volMin, volMax, min_s = 3, max_s = 15):
    info = info.fillna(0)
    return min_s + (info - volMin) / (volMax - volMin) * (max_s - min_s)

def plotInfo(info, min, max, symb, col, name, threshold):
    mask = info['volume'] > threshold
    fig.add_trace(go.Scatter(
        x=info.timestamp[mask], y=info.price[mask],
        mode='markers', marker=dict(symbol=symb, color=col, size=scale_vol(info.volume[mask], min, max)),
        name=name
    ))

plotInfo(bids, min_bid, max_bid, "circle", "blue", "Bids", volume_threshold)
plotInfo(asks, min_ask, max_ask, "circle", "red", "asks", volume_threshold)

best_bids = bids.loc[bids.groupby('timestamp')['volume'].idxmax()][['timestamp', 'price']].rename(columns={'price': 'bid_price'})
best_asks = asks.loc[asks.groupby('timestamp')['volume'].idxmax()][['timestamp', 'price']].rename(columns={'price': 'ask_price'})

# Merge on timestamp and compute average
accPrice = best_bids.merge(best_asks, on='timestamp')
accPrice['price'] = (accPrice['bid_price'] + accPrice['ask_price']) / 2

fig.add_trace(go.Scatter(
    x=accPrice['timestamp'], y=accPrice['price'],
    mode='lines', line=dict(color='grey', width=1),
    name='Mid'
))

if has_trading_history:
    with open('tradingHistory.json', 'r') as f:
        trade_data = json.load(f)

    trades_df = (pd.DataFrame(trade_data['tradeHistory'])
                   .rename(columns={'quantity': 'volume'})
                   .query(f"symbol == '{product}'"))

    buys = trades_df[trades_df['buyer'] == 'SUBMISSION']
    sells = trades_df[trades_df['seller'] == 'SUBMISSION']

    max_quantity = max(buys.volume.max(), sells.volume.max())
    min_quantity = min(buys.volume.min(), sells.volume.min())

    plotInfo(buys, min_quantity, max_quantity, 'triangle-up', 'green', 'Buys', 0)
    plotInfo(sells, min_quantity, max_quantity, 'triangle-down', 'red', 'Sells', 0)

fig.update_layout(
    title=f'{product} Orderbook over Time',
    xaxis_title='Timestamp',
    yaxis_title='Price',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black')
)

fig.show()

# --- PnL Plot ---
# Merge trades with accPrice to get mark-to-market price at each timestamp
if has_trading_history:
    all_trades = trades_df[(trades_df['buyer'] == 'SUBMISSION') | (trades_df['seller'] == 'SUBMISSION')].copy()
    all_trades['signed_qty'] = all_trades.apply(
        lambda r: r['volume'] if r['buyer'] == 'SUBMISSION' else -r['volume'], axis=1
    )

    # Compute running position and cash
    all_trades = all_trades.sort_values('timestamp')
    all_trades['position'] = all_trades['signed_qty'].cumsum()
    all_trades['cash'] = (-all_trades['signed_qty'] * all_trades['price']).cumsum()

    # Merge with mid price to mark open position to market
    pnl = all_trades.merge(accPrice[['timestamp', 'price']], on='timestamp', how='left', suffixes=('_trade', '_mid'))
    pnl['price_mid'] = pnl['price_mid'].ffill()  # fill gaps with last known mid
    pnl['pnl'] = pnl['cash'] + pnl['position'] * pnl['price_mid']

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=pnl['timestamp'], y=pnl['pnl'],
        mode='lines+markers',
        line=dict(color='purple', width=2),
        marker=dict(size=6),
        name='PnL'
    ))
    fig2.add_hline(y=0, line_dash='dash', line_color='grey')

    fig2.update_layout(
        title=f'{product} Approximated PnL over Time',
        xaxis_title='Timestamp',
        yaxis_title='PnL (SeaShells)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        xaxis=dict(gridcolor='lightgrey'),
        yaxis=dict(gridcolor='lightgrey')
    )

    fig2.show()