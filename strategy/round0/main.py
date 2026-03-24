from __future__ import annotations

import json
from typing import Any, Dict, List, cast


class Order:
    def __init__(self, symbol: str, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity


class Trader:
    POSITION_LIMITS: Dict[str, int] = {
        "EMERALDS": 20,
        "TOMATOES": 20,
    }

    HISTORY_LEN = 60
    EMERALDS_TAKE_EDGE = 2.0
    TOMATOES_TAKE_EDGE = 3.0
    EMERALDS_INVENTORY_PENALTY = 0.20
    TOMATOES_INVENTORY_PENALTY = 0.25
    INVENTORY_EDGE_REDUCTION_START = 15
    INVENTORY_EDGE_REDUCTION = 1.0
    EOD_FLATTEN_START_TS = 980000
    EOD_FLATTEN_SHIFT = 1.5

    def run(self, state: Any):
        memory = self._load_memory(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            if product not in self.POSITION_LIMITS:
                continue

            position = state.position.get(product, 0)
            orders = self._trade_product(
                product=product,
                order_depth=order_depth,
                position=position,
                memory=memory,
                timestamp=getattr(state, "timestamp", 0),
            )
            result[product] = orders

        trader_data = self._dump_memory(memory)
        conversions = 0
        return result, conversions, trader_data

    def _trade_product(
        self,
        product: str,
        order_depth: Any,
        position: int,
        memory: Dict[str, Any],
        timestamp: int,
    ) -> List[Order]:
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        orders: List[Order] = []

        best_bid = max(buy_orders.keys()) if buy_orders else None
        best_ask = min(sell_orders.keys()) if sell_orders else None

        if best_bid is not None and best_ask is not None:
            mid = 0.5 * (best_bid + best_ask)
        elif best_bid is not None:
            mid = float(best_bid)
        elif best_ask is not None:
            mid = float(best_ask)
        else:
            return orders

        self._update_history(memory, product, mid)

        fair_value = self._fair_value(product, memory, mid)

        limit = self.POSITION_LIMITS[product]
        remaining_buy = max(0, limit - position)
        remaining_sell = max(0, limit + position)

        if product == "EMERALDS":
            take_edge = self.EMERALDS_TAKE_EDGE
            inventory_penalty = self.EMERALDS_INVENTORY_PENALTY
        else:
            take_edge = self.TOMATOES_TAKE_EDGE
            inventory_penalty = self.TOMATOES_INVENTORY_PENALTY

        fair_with_inventory = fair_value - inventory_penalty * position

        if abs(position) >= self.INVENTORY_EDGE_REDUCTION_START:
            take_edge = max(1.0, take_edge - self.INVENTORY_EDGE_REDUCTION)

        if timestamp >= self.EOD_FLATTEN_START_TS:
            if position > 0:
                fair_with_inventory -= self.EOD_FLATTEN_SHIFT
                take_edge = 0.0
            elif position < 0:
                fair_with_inventory += self.EOD_FLATTEN_SHIFT
                take_edge = 0.0

        if sell_orders and remaining_buy > 0:
            for ask_price in sorted(sell_orders.keys()):
                if remaining_buy <= 0:
                    break
                if ask_price > fair_with_inventory - take_edge:
                    break
                ask_volume = -sell_orders[ask_price]
                take_qty = min(remaining_buy, ask_volume)
                if take_qty > 0:
                    orders.append(Order(product, ask_price, take_qty))
                    remaining_buy -= take_qty

        if buy_orders and remaining_sell > 0:
            for bid_price in sorted(buy_orders.keys(), reverse=True):
                if remaining_sell <= 0:
                    break
                if bid_price < fair_with_inventory + take_edge:
                    break
                bid_volume = buy_orders[bid_price]
                take_qty = min(remaining_sell, bid_volume)
                if take_qty > 0:
                    orders.append(Order(product, bid_price, -take_qty))
                    remaining_sell -= take_qty

        return orders

    def _fair_value(self, product: str, memory: Dict[str, Any], mid: float) -> float:
        history = memory.setdefault("mid_history", {}).setdefault(product, [])

        if product == "EMERALDS":
            if len(history) < 2:
                return 10000.0
            recent_drift = history[-1] - history[-2]
            return 10000.0 - 0.20 * recent_drift

        fast_window = history[-8:] if len(history) >= 8 else history
        slow_window = history[-30:] if len(history) >= 30 else history

        fast_mean = sum(fast_window) / len(fast_window)
        slow_mean = sum(slow_window) / len(slow_window)

        if len(history) >= 2:
            last_return = history[-1] - history[-2]
        else:
            last_return = 0.0

        fair = 0.65 * fast_mean + 0.35 * slow_mean - 0.55 * last_return
        return fair if fair == fair else mid

    def _update_history(self, memory: Dict[str, Any], product: str, mid: float) -> None:
        history = memory.setdefault("mid_history", {}).setdefault(product, [])
        history.append(mid)
        if len(history) > self.HISTORY_LEN:
            del history[: len(history) - self.HISTORY_LEN]

    def _load_memory(self, trader_data: str) -> Dict[str, Any]:
        if not trader_data:
            return {"mid_history": {}}
        try:
            parsed_obj: Any = json.loads(trader_data)
            if isinstance(parsed_obj, dict):
                parsed = cast(Dict[str, Any], parsed_obj)
                if not isinstance(parsed.get("mid_history"), dict):
                    parsed["mid_history"] = {}
                return parsed
            return {"mid_history": {}}
        except Exception:
            return {"mid_history": {}}

    def _dump_memory(self, memory: Dict[str, Any]) -> str:
        return json.dumps(memory, separators=(",", ":"))
