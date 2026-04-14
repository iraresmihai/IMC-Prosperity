import json

from datamodel import OrderDepth, TradingState, Order
from typing import List


class Trader:

    def findRealPrice(self, orders: OrderDepth) -> float:
        buy_price = max(orders.buy_orders, key=lambda o: orders.buy_orders[o])
        sell_price = max(orders.sell_orders, key=lambda o: orders.sell_orders[o])

        return (buy_price + sell_price) / 2

    def findBestOffers(self, orders: OrderDepth):
        return {
            "best_sell": max(orders.buy_orders) if orders.buy_orders else 0,
            "best_buy": min(orders.sell_orders) if orders.sell_orders else 1000000000
        }

    def run(self, state: TradingState):
        result = {}
        #to make a profit => skew should be at least 2x our skew
        skew_threshold = 20 # skew of big volume bot is 10, there are bots with less skew => aim to buy and sell at smaller skew
        our_skew = 9 #this is the smaller skew
        min_n = 50

        t = state.timestamp
        if state.traderData:
            data = json.loads(state.traderData)
            n = data.get("n", 0)
            sum_x = data.get("sum_x", 0)
            sum_y = data.get("sum_y", 0)
            sum_xx = data.get("sum_xx", 0)
            sum_xy = data.get("sum_xy", 0)
        else:
            n = sum_x = sum_y = sum_xx = sum_xy = 0
        # Standard position limit for early rounds
        POSITION_LIMIT = 20

        for product in state.order_depths:
            position = state.position.get(product, 0)
            new_orders = []
            if product == "INTARIAN_PEPPER_ROOT":
                all_orders: OrderDepth = state.order_depths[product]
                order_info = self.findBestOffers(all_orders)
                mid_price = self.findRealPrice(all_orders)
                n += 1
                sum_x += t
                sum_y += mid_price
                sum_xx += t * t
                sum_xy += mid_price * t

                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
                intercept = (sum_y - slope * sum_x) / n
                fair_value = slope * t + intercept

                skew = mid_price - fair_value
                if n < min_n:
                    if position != POSITION_LIMIT:
                        new_orders.append(Order(product, min(order_info["best_buy"], int(mid_price + our_skew)), POSITION_LIMIT - position))
                else:
                    if skew <= -skew_threshold: #we buy as much as possible
                        to_long = POSITION_LIMIT - position
                        new_orders.append(Order(product, min(order_info["best_buy"], int(mid_price + our_skew)), to_long))
                    else:
                        if skew >= skew_threshold: #we sell as much as possible
                            to_short = -position - POSITION_LIMIT
                            new_orders.append(Order(product, max(order_info["best_sell"], int(mid_price - our_skew)), to_short))
                        else: # if we are in normal boundaries, we would like to just at least delete a possible short position
                            if skew <= our_skew and position < 0:
                                to_long = -position
                                new_orders.append(Order(product, min(order_info["best_buy"], int(mid_price + our_skew)), to_long))
                                if skew <= 0: #if skew is negative, we give an order but not very big
                                    new_orders.append(Order(product, min(order_info["best_buy"], int(mid_price - our_skew)), min(POSITION_LIMIT - position, int(POSITION_LIMIT / 2))))
            result[product] = new_orders
            '''
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Get current inventory
            current_position = state.position.get(product, 0)

            if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:

                # Get the current top of the order book
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

                # Calculate how much we are allowed to trade
                acceptable_buy_volume = POSITION_LIMIT - current_position
                acceptable_sell_volume = -POSITION_LIMIT - current_position

                # --- INVENTORY SKEWING ---
                # If we have a lot of long inventory (> 10), we stop being aggressive on buys.
                # If we have a lot of short inventory (< -10), we stop being aggressive on sells.
                buy_aggressiveness = 1 if current_position < 10 else 0
                sell_aggressiveness = 1 if current_position > -10 else 0

                # --- PENNYING ---
                # Calculate the spread. If it's larger than 2, we have room to "penny" (jump the queue)
                # without accidentally crossing our own orders.
                spread = best_ask - best_bid

                if spread > 2:
                    our_bid = best_bid + buy_aggressiveness
                    our_ask = best_ask - sell_aggressiveness
                else:
                    our_bid = best_bid
                    our_ask = best_ask

                # --- PLACE ORDERS ---
                if acceptable_buy_volume > 0:
                    orders.append(Order(product, our_bid, acceptable_buy_volume))

                if acceptable_sell_volume < 0:
                    orders.append(Order(product, our_ask, acceptable_sell_volume))

            result[product] = orders
            '''
        # Update metadata for the logs
        state = json.dumps({
            "n": n,
            "sum_x": sum_x,
            "sum_y": sum_y,
            "sum_xx": sum_xx,
            "sum_xy": sum_xy
        })

        traderData = state
        conversions = 1

        return result, conversions, traderData