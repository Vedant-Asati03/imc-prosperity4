from __future__ import annotations

from datamodel import Order, TradingState


class Trader:
    POSITION_LIMITS: dict[str, int] = {}

    def bid(self) -> int:
        return 15

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        result: dict[str, list[Order]] = {}
        for product in state.order_depths:
            result[product] = []

        conversions = 0
        trader_data = state.traderData
        return result, conversions, trader_data
