from __future__ import annotations

from datamodel import Order, OrderDepth, TradingState

EMERALDS_SYMBOL = "EMERALDS"
TOMATOES_SYMBOL = "TOMATOES"


class ProductTrader:
    def __init__(self, symbol: str, state: TradingState, position_limit: int):
        self.symbol = symbol
        self.state = state
        self.position_limit = position_limit
        self.position = state.position.get(symbol, 0)
        self.orders: list[Order] = []

        self.max_buy_volume = self.position_limit - self.position
        self.max_sell_volume = self.position_limit + self.position

        depth: OrderDepth | None = state.order_depths.get(symbol)
        if depth is None:
            self.buy_orders: dict[int, int] = {}
            self.sell_orders: dict[int, int] = {}
        else:
            self.buy_orders = {
                price: abs(volume)
                for price, volume in sorted(depth.buy_orders.items(), reverse=True)
            }
            self.sell_orders = {
                price: abs(volume)
                for price, volume in sorted(depth.sell_orders.items())
            }

        self.best_bid = max(self.buy_orders.keys()) if self.buy_orders else None
        self.best_ask = min(self.sell_orders.keys()) if self.sell_orders else None
        self.outer_bid = min(self.buy_orders.keys()) if self.buy_orders else None
        self.outer_ask = max(self.sell_orders.keys()) if self.sell_orders else None

        self.wall_mid = None
        if self.outer_bid is not None and self.outer_ask is not None:
            self.wall_mid = (self.outer_bid + self.outer_ask) / 2

    def buy(self, price: int, volume: int) -> None:
        clipped_volume = min(abs(int(volume)), self.max_buy_volume)
        if clipped_volume <= 0:
            return
        self.orders.append(Order(self.symbol, int(price), clipped_volume))
        self.max_buy_volume -= clipped_volume

    def sell(self, price: int, volume: int) -> None:
        clipped_volume = min(abs(int(volume)), self.max_sell_volume)
        if clipped_volume <= 0:
            return
        self.orders.append(Order(self.symbol, int(price), -clipped_volume))
        self.max_sell_volume -= clipped_volume


class EmeraldsTrader(ProductTrader):
    def build_orders(self) -> list[Order]:
        if self.wall_mid is None or self.outer_bid is None or self.outer_ask is None:
            return self.orders

        for ask_price, ask_volume in self.sell_orders.items():
            if ask_price <= self.wall_mid - 1:
                self.buy(ask_price, ask_volume)
            elif ask_price <= self.wall_mid and self.position < 0:
                self.buy(ask_price, min(ask_volume, abs(self.position)))

        for bid_price, bid_volume in self.buy_orders.items():
            if bid_price >= self.wall_mid + 1:
                self.sell(bid_price, bid_volume)
            elif bid_price >= self.wall_mid and self.position > 0:
                self.sell(bid_price, min(bid_volume, self.position))

        bid_price = int(self.outer_bid + 1)
        ask_price = int(self.outer_ask - 1)

        for level_price, level_volume in self.buy_orders.items():
            if level_price < self.wall_mid:
                candidate = level_price + 1 if level_volume > 1 else level_price
                bid_price = max(bid_price, candidate)
                break

        for level_price, level_volume in self.sell_orders.items():
            if level_price > self.wall_mid:
                candidate = level_price - 1 if level_volume > 1 else level_price
                ask_price = min(ask_price, candidate)
                break

        self.buy(bid_price, self.max_buy_volume)
        self.sell(ask_price, self.max_sell_volume)
        return self.orders


class TomatoesTrader(ProductTrader):
    def build_orders(self) -> list[Order]:
        if self.wall_mid is None or self.outer_bid is None or self.outer_ask is None:
            return self.orders

        for ask_price, ask_volume in self.sell_orders.items():
            if ask_price < self.wall_mid:
                self.buy(ask_price, ask_volume)

        for bid_price, bid_volume in self.buy_orders.items():
            if bid_price > self.wall_mid:
                self.sell(bid_price, bid_volume)

        bid_price = int(self.outer_bid + 1)
        ask_price = int(self.outer_ask - 1)

        self.buy(bid_price, self.max_buy_volume)
        self.sell(ask_price, self.max_sell_volume)
        return self.orders


class Trader:
    POSITION_LIMITS: dict[str, int] = {
        EMERALDS_SYMBOL: 80,
        TOMATOES_SYMBOL: 80,
    }

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        result: dict[str, list[Order]] = {}

        if EMERALDS_SYMBOL in state.order_depths:
            emeralds = EmeraldsTrader(
                symbol=EMERALDS_SYMBOL,
                state=state,
                position_limit=self.POSITION_LIMITS[EMERALDS_SYMBOL],
            )
            result[EMERALDS_SYMBOL] = emeralds.build_orders()

        if TOMATOES_SYMBOL in state.order_depths:
            tomatoes = TomatoesTrader(
                symbol=TOMATOES_SYMBOL,
                state=state,
                position_limit=self.POSITION_LIMITS[TOMATOES_SYMBOL],
            )
            result[TOMATOES_SYMBOL] = tomatoes.build_orders()

        conversions = 0
        trader_data = state.traderData
        return result, conversions, trader_data
