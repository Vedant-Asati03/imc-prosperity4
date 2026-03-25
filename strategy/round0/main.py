from __future__ import annotations

from typing import Any, cast

import jsonpickle  # type: ignore

from datamodel import Order, OrderDepth, TradingState


class Product:
    EMERALDS = "EMERALDS"
    TOMATOES = "TOMATOES"


PARAMS: dict[str, dict[str, Any]] = {
    Product.EMERALDS: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 1,
        "soft_position_limit": 8,
    },
    Product.TOMATOES: {
        "take_width": 0,
        "clear_width": 0,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "reversion_beta": -0.25,
        "disregard_edge": 0,
        "join_edge": 0,
        "default_edge": 2,
    },
}


class Trader:
    POSITION_LIMITS: dict[str, int] = {
        Product.EMERALDS: 20,
        Product.TOMATOES: 20,
    }

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or PARAMS
        self.LIMIT = dict(self.POSITION_LIMITS)

    def take_best_orders(
        self,
        product: str,
        fair_value: float,
        take_width: float,
        orders: list[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> tuple[int, int]:
        position_limit = self.LIMIT[product]

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]

            if (not prevent_adverse or abs(best_ask_amount) <= adverse_volume) and (
                best_ask <= fair_value - take_width
            ):
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if (not prevent_adverse or abs(best_bid_amount) <= adverse_volume) and (
                best_bid >= fair_value + take_width
            ):
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_order_volume += quantity

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: list[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[int, int]:
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))

        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: list[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[int, int]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def tomatoes_fair_value(
        self, order_depth: OrderDepth, trader_object: dict[str, Any]
    ) -> float | None:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        filtered_ask = [
            price
            for price in order_depth.sell_orders.keys()
            if abs(order_depth.sell_orders[price])
            >= self.params[Product.TOMATOES]["adverse_volume"]
        ]
        filtered_bid = [
            price
            for price in order_depth.buy_orders.keys()
            if abs(order_depth.buy_orders[price])
            >= self.params[Product.TOMATOES]["adverse_volume"]
        ]

        mm_ask = min(filtered_ask) if filtered_ask else None
        mm_bid = max(filtered_bid) if filtered_bid else None

        if mm_ask is None or mm_bid is None:
            mmmid_price = trader_object.get(
                "tomatoes_last_price", (best_ask + best_bid) / 2
            )
        else:
            mmmid_price = (mm_ask + mm_bid) / 2

        if trader_object.get("tomatoes_last_price") is not None:
            last_price = trader_object["tomatoes_last_price"]
            last_returns = (mmmid_price - last_price) / last_price
            pred_returns = (
                last_returns * self.params[Product.TOMATOES]["reversion_beta"]
            )
            fair = mmmid_price + (mmmid_price * pred_returns)
        else:
            fair = mmmid_price

        trader_object["tomatoes_last_price"] = mmmid_price
        return fair

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> tuple[list[Order], int, int]:
        orders: list[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[list[Order], int, int]:
        orders: list[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ) -> tuple[list[Order], int, int]:
        orders: list[Order] = []

        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            ask = (
                best_ask_above_fair
                if abs(best_ask_above_fair - fair_value) <= join_edge
                else best_ask_above_fair - 1
            )

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            bid = (
                best_bid_below_fair
                if abs(fair_value - best_bid_below_fair) <= join_edge
                else best_bid_below_fair + 1
            )

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        trader_object: dict[str, Any] = {}
        if state.traderData:
            try:
                decoded = jsonpickle.decode(state.traderData) # type: ignore
                if isinstance(decoded, dict):
                    trader_object = cast(dict[str, Any], decoded)
            except Exception:
                trader_object = {}

        result: dict[str, list[Order]] = {}

        if Product.EMERALDS in self.params and Product.EMERALDS in state.order_depths:
            position = state.position.get(Product.EMERALDS, 0)
            take_orders, buy_vol, sell_vol = self.take_orders(
                Product.EMERALDS,
                state.order_depths[Product.EMERALDS],
                self.params[Product.EMERALDS]["fair_value"],
                self.params[Product.EMERALDS]["take_width"],
                position,
            )
            clear_orders, buy_vol, sell_vol = self.clear_orders(
                Product.EMERALDS,
                state.order_depths[Product.EMERALDS],
                self.params[Product.EMERALDS]["fair_value"],
                self.params[Product.EMERALDS]["clear_width"],
                position,
                buy_vol,
                sell_vol,
            )
            make_orders, _, _ = self.make_orders(
                Product.EMERALDS,
                state.order_depths[Product.EMERALDS],
                self.params[Product.EMERALDS]["fair_value"],
                position,
                buy_vol,
                sell_vol,
                self.params[Product.EMERALDS]["disregard_edge"],
                self.params[Product.EMERALDS]["join_edge"],
                self.params[Product.EMERALDS]["default_edge"],
                True,
                self.params[Product.EMERALDS]["soft_position_limit"],
            )
            result[Product.EMERALDS] = take_orders + clear_orders + make_orders

        if Product.TOMATOES in self.params and Product.TOMATOES in state.order_depths:
            position = state.position.get(Product.TOMATOES, 0)
            fair_value = self.tomatoes_fair_value(
                state.order_depths[Product.TOMATOES], trader_object
            )
            if fair_value is not None:
                take_orders, buy_vol, sell_vol = self.take_orders(
                    Product.TOMATOES,
                    state.order_depths[Product.TOMATOES],
                    fair_value,
                    self.params[Product.TOMATOES]["take_width"],
                    position,
                    self.params[Product.TOMATOES]["prevent_adverse"],
                    self.params[Product.TOMATOES]["adverse_volume"],
                )
                clear_orders, buy_vol, sell_vol = self.clear_orders(
                    Product.TOMATOES,
                    state.order_depths[Product.TOMATOES],
                    fair_value,
                    self.params[Product.TOMATOES]["clear_width"],
                    position,
                    buy_vol,
                    sell_vol,
                )
                make_orders, _, _ = self.make_orders(
                    Product.TOMATOES,
                    state.order_depths[Product.TOMATOES],
                    fair_value,
                    position,
                    buy_vol,
                    sell_vol,
                    self.params[Product.TOMATOES]["disregard_edge"],
                    self.params[Product.TOMATOES]["join_edge"],
                    self.params[Product.TOMATOES]["default_edge"],
                )
                result[Product.TOMATOES] = take_orders + clear_orders + make_orders

        conversions = 0
        encoded = jsonpickle.encode(trader_object) # type: ignore
        trader_data = encoded if isinstance(encoded, str) else ""
        return result, conversions, trader_data
