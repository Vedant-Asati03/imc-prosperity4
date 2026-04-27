from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd

from datamodel import (
    OrderDepth,
    ReplayFrame,
    Trade,
    build_replay_frames,
    discover_round_days,
)

MAX_RUN_TIME_MS = 900
MAX_TRADER_DATA_LEN = 50_000

logs_dir = Path("logs")
plots_dir = Path("plots")

logs_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class FillEvent:
    timestamp: int
    product: str
    side: str
    price: int
    quantity: int


@dataclass
class RunnerIssue:
    severity: str
    message: str
    timestamp: int | None = None


class RunnerValidationError(Exception):
    def __init__(self, issues: List[RunnerIssue]) -> None:
        self.issues = issues
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = ["Local runner validation failed:"]
        for issue in self.issues:
            ts = f" [ts={issue.timestamp}]" if issue.timestamp is not None else ""
            lines.append(f"- {issue.severity.upper()}{ts}: {issue.message}")
        return "\n".join(lines)


def has_errors(issues: List[RunnerIssue]) -> bool:
    return any(issue.severity == "error" for issue in issues)


def generate_run_id() -> str:
    return str(int(time.time() * 1000) % 100000)


def load_trader_class(strategy_file: Path) -> type[Any]:
    strategy_file = strategy_file.resolve()
    if not strategy_file.exists():
        raise FileNotFoundError(f"Strategy file not found: {strategy_file}")

    module_name = f"strategy_submission_{strategy_file.stem}_{int(time.time() * 1e6)}"
    spec = importlib.util.spec_from_file_location(module_name, strategy_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load strategy module from {strategy_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    trader_cls = getattr(module, "Trader", None)
    if trader_cls is None:
        raise AttributeError(
            f"Strategy file {strategy_file} does not define Trader class"
        )

    return trader_cls


def check_position_limit(position: int, orders: List[Any], limit: int) -> bool:
    total_buy = sum(order.quantity for order in orders if order.quantity > 0)
    total_sell = -sum(order.quantity for order in orders if order.quantity < 0)
    return (position + total_buy <= limit) and (position - total_sell >= -limit)


def execute_orders(
    timestamp: int,
    orders_by_product: Dict[str, List[Any]],
    snapshot: Dict[str, OrderDepth],
    position: Dict[str, int],
    cash: float,
    limits: Dict[str, int],
) -> Tuple[float, int, int, List[FillEvent], List[str], Dict[str, List[Trade]]]:
    fills = 0
    volume = 0
    fill_events: List[FillEvent] = []
    rejected_products: List[str] = []
    own_trades_next: Dict[str, List[Trade]] = defaultdict(list)

    for product, orders in orders_by_product.items():
        if product not in snapshot:
            continue

        limit = limits.get(product, 20)
        current_pos = position.get(product, 0)

        if not check_position_limit(current_pos, orders, limit):
            rejected_products.append(product)
            continue

        depth = snapshot[product]

        for order in orders:
            qty_remaining = int(order.quantity)

            if qty_remaining > 0:
                for ask_price in sorted(depth.sell_orders.keys()):
                    if qty_remaining <= 0:
                        break
                    if int(order.price) < ask_price:
                        break

                    available = -depth.sell_orders[ask_price]
                    traded = min(qty_remaining, available)
                    if traded <= 0:
                        continue

                    depth.sell_orders[ask_price] += traded
                    qty_remaining -= traded
                    position[product] = position.get(product, 0) + traded
                    cash -= ask_price * traded
                    fills += 1
                    volume += traded
                    fill_events.append(
                        FillEvent(
                            timestamp=timestamp,
                            product=product,
                            side="BUY",
                            price=ask_price,
                            quantity=traded,
                        )
                    )
                    own_trades_next[product].append(
                        Trade(
                            symbol=product,
                            price=ask_price,
                            quantity=traded,
                            buyer="SUBMISSION",
                            seller="",
                            timestamp=timestamp,
                        )
                    )

            elif qty_remaining < 0:
                qty_to_sell = -qty_remaining
                for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                    if qty_to_sell <= 0:
                        break
                    if int(order.price) > bid_price:
                        break

                    available = depth.buy_orders[bid_price]
                    traded = min(qty_to_sell, available)
                    if traded <= 0:
                        continue

                    depth.buy_orders[bid_price] -= traded
                    qty_to_sell -= traded
                    position[product] = position.get(product, 0) - traded
                    cash += bid_price * traded
                    fills += 1
                    volume += traded
                    fill_events.append(
                        FillEvent(
                            timestamp=timestamp,
                            product=product,
                            side="SELL",
                            price=bid_price,
                            quantity=traded,
                        )
                    )
                    own_trades_next[product].append(
                        Trade(
                            symbol=product,
                            price=bid_price,
                            quantity=traded,
                            buyer="",
                            seller="SUBMISSION",
                            timestamp=timestamp,
                        )
                    )

    return cash, fills, volume, fill_events, rejected_products, dict(own_trades_next)


def mark_to_market(
    cash: float, position: Dict[str, int], mids: Dict[str, float]
) -> float:
    value = cash
    for product, qty in position.items():
        value += qty * mids.get(product, 0.0)
    return value


def forced_flatten_value(
    position: Dict[str, int], snapshot: Dict[str, OrderDepth]
) -> float:
    value = 0.0
    for product, qty in position.items():
        if qty == 0 or product not in snapshot:
            continue
        depth = snapshot[product]
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None

        if qty > 0:
            px = best_bid if best_bid is not None else best_ask
            if px is not None:
                value += qty * px
        else:
            px = best_ask if best_ask is not None else best_bid
            if px is not None:
                value += qty * px
    return value


def run_day(
    day: int,
    frames: List[ReplayFrame],
    trader_cls: type[Any],
) -> Dict[str, Any]:
    if not frames:
        raise ValueError(f"No replay frames found for day {day}")

    trader = trader_cls()
    products = sorted(frames[0].state.listings.keys())
    limits: Dict[str, int] = dict(getattr(trader, "POSITION_LIMITS", {}))
    missing_limits = [product for product in products if product not in limits]
    for product in missing_limits:
        limits[product] = 20

    position: Dict[str, int] = {product: 0 for product in products}
    trader_data = ""
    own_trades_prev: Dict[str, List[Trade]] = {product: [] for product in products}

    cash = 0.0
    total_fills = 0
    total_volume = 0
    warnings_log: List[RunnerIssue] = []
    if missing_limits:
        warnings_log.append(
            RunnerIssue(
                severity="warning",
                message=(
                    "POSITION_LIMITS missing for product(s): "
                    f"{', '.join(missing_limits)}; using default limit 20"
                ),
            )
        )
    timeline: Dict[str, Any] = {
        "timestamp": [],
        "cash": [],
        "pnl": [],
        "mid_by_product": {product: [] for product in products},
        "pos_by_product": {product: [] for product in products},
    }

    final_snapshot: Dict[str, OrderDepth] = {}

    for frame in frames:
        timestamp = frame.timestamp

        state = frame.state
        state.traderData = trader_data
        state.position = position.copy()
        state.own_trades = {
            product: list(own_trades_prev.get(product, [])) for product in products
        }

        issues: List[RunnerIssue] = []
        orders_by_product: Dict[str, Any] = {}
        conversions: Any = 0
        trader_data_out: Any = ""

        start = time.perf_counter()
        try:
            raw_result_obj: Any = trader.run(state)
        except Exception as exc:
            raise RunnerValidationError(
                [
                    RunnerIssue(
                        severity="error",
                        timestamp=timestamp,
                        message=f"Trader.run raised {type(exc).__name__}: {exc}",
                    )
                ]
            ) from exc

        if not isinstance(raw_result_obj, tuple):
            issues.append(
                RunnerIssue(
                    severity="error",
                    timestamp=timestamp,
                    message=(
                        "Trader.run must return a tuple of length 3: "
                        "(orders_by_product, conversions, trader_data)"
                    ),
                )
            )
        else:
            raw_result = cast(tuple[Any, ...], raw_result_obj)
            if len(raw_result) != 3:
                issues.append(
                    RunnerIssue(
                        severity="error",
                        timestamp=timestamp,
                        message=(
                            "Trader.run must return a tuple of length 3: "
                            "(orders_by_product, conversions, trader_data)"
                        ),
                    )
                )
            else:
                orders_raw, conversions, trader_data_out = raw_result
                if isinstance(orders_raw, dict):
                    orders_by_product = cast(Dict[str, Any], orders_raw)
                else:
                    issues.append(
                        RunnerIssue(
                            severity="error",
                            timestamp=timestamp,
                            message="orders_by_product must be a dict[str, list[Order]]",
                        )
                    )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms > MAX_RUN_TIME_MS:
            issues.append(
                RunnerIssue(
                    severity="error",
                    timestamp=timestamp,
                    message=f"Trader.run exceeded {MAX_RUN_TIME_MS}ms: {elapsed_ms:.2f}ms",
                )
            )

        trader_data = trader_data_out
        if not isinstance(trader_data, str):
            issues.append(
                RunnerIssue(
                    severity="warning",
                    timestamp=timestamp,
                    message="traderData is not a string; coercing to string",
                )
            )
            trader_data = str(trader_data)

        if len(trader_data) > MAX_TRADER_DATA_LEN:
            issues.append(
                RunnerIssue(
                    severity="warning",
                    timestamp=timestamp,
                    message=(
                        f"traderData length {len(trader_data)} exceeds "
                        f"{MAX_TRADER_DATA_LEN}; truncating"
                    ),
                )
            )
            trader_data = trader_data[:MAX_TRADER_DATA_LEN]

        if conversions not in (0, None):
            issues.append(
                RunnerIssue(
                    severity="error",
                    timestamp=timestamp,
                    message=(
                        "This local runner does not simulate conversions. "
                        "Trader must return conversions=0 or None."
                    ),
                )
            )

        for product_key, product_orders in orders_by_product.items():
            product = str(product_key)
            if product not in state.order_depths:
                issues.append(
                    RunnerIssue(
                        severity="error",
                        timestamp=timestamp,
                        message=f"Unknown product in orders: {product}",
                    )
                )
                continue

            if not isinstance(product_orders, list):
                issues.append(
                    RunnerIssue(
                        severity="error",
                        timestamp=timestamp,
                        message=f"Orders for {product} must be a list",
                    )
                )
                continue

            for order in cast(List[Any], product_orders):
                for field in ("symbol", "price", "quantity"):
                    if not hasattr(order, field):
                        issues.append(
                            RunnerIssue(
                                severity="error",
                                timestamp=timestamp,
                                message=(
                                    f"Malformed order for {product}: missing field '{field}'"
                                ),
                            )
                        )
                        break
                else:
                    symbol = str(getattr(order, "symbol"))
                    if symbol != product:
                        issues.append(
                            RunnerIssue(
                                severity="error",
                                timestamp=timestamp,
                                message=(
                                    f"Order symbol mismatch: key '{product}' "
                                    f"vs order.symbol '{symbol}'"
                                ),
                            )
                        )

        warnings_log.extend(issue for issue in issues if issue.severity == "warning")
        if has_errors(issues):
            raise RunnerValidationError(issues)

        snapshot: Dict[str, OrderDepth] = {}
        for product, depth in state.order_depths.items():
            copied_depth = OrderDepth()
            copied_depth.buy_orders = dict(depth.buy_orders)
            copied_depth.sell_orders = dict(depth.sell_orders)
            snapshot[product] = copied_depth

        cash, fills, volume, _, _, own_trades_next = execute_orders(
            timestamp=timestamp,
            orders_by_product=cast(Dict[str, List[Any]], orders_by_product),
            snapshot=snapshot,
            position=position,
            cash=cash,
            limits=limits,
        )
        own_trades_prev = {
            product: own_trades_next.get(product, []) for product in products
        }

        total_fills += fills
        total_volume += volume

        pnl_now = mark_to_market(cash, position, frame.mid_prices)
        timeline["timestamp"].append(timestamp)
        timeline["cash"].append(cash)
        timeline["pnl"].append(pnl_now)
        for product in products:
            timeline["mid_by_product"][product].append(
                frame.mid_prices.get(product, 0.0)
            )
            timeline["pos_by_product"][product].append(position.get(product, 0))

        final_snapshot = snapshot

    final_ts = frames[-1].timestamp
    final_pnl = mark_to_market(cash, position, frames[-1].mid_prices)
    flatten_pnl = cash + forced_flatten_value(position, final_snapshot)

    return {
        "day": day,
        "ticks": len(frames),
        "final_timestamp": final_ts,
        "products": products,
        "pnl": final_pnl,
        "cash": cash,
        "position": dict(position),
        "fills": total_fills,
        "volume": total_volume,
        "forced_flatten_pnl": flatten_pnl,
        "warnings": [
            {
                "severity": w.severity,
                "timestamp": w.timestamp,
                "message": w.message,
            }
            for w in warnings_log
        ],
        "timeline": timeline,
    }


def plot_day_result(day_result: Dict[str, Any], output_dir: Path, label: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    timeline = day_result["timeline"]
    ts = timeline["timestamp"]
    products: List[str] = day_result["products"]

    plot_module = cast(Any, plt)
    fig, axes = plot_module.subplots(3, 1, figsize=(14, 10), sharex=True)

    for product in products:
        axes[0].plot(
            ts,
            timeline["mid_by_product"][product],
            label=f"{product} mid",
            linewidth=1.2,
        )
    axes[0].set_ylabel("Mid Price")
    axes[0].set_title(label)
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    for product in products:
        axes[1].plot(
            ts,
            timeline["pos_by_product"][product],
            label=f"{product} pos",
            linewidth=1.2,
        )
    axes[1].set_ylabel("Position")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.25)

    axes[2].plot(ts, timeline["pnl"], label="Mark-to-Market PnL", linewidth=1.4)
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    axes[2].set_ylabel("PnL")
    axes[2].set_xlabel("Timestamp")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    out_path = output_dir / f"{label}.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return out_path


def summarize_product_day(
    prices_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> Dict[str, Dict[str, float | int | None]]:
    summaries: Dict[str, Dict[str, float | int | None]] = {}

    if prices_df.empty:
        return summaries

    safe_prices_df = prices_df.copy()
    col = safe_prices_df.get("mid_price")
    if col is None:
        col = pd.Series([None] * len(safe_prices_df), index=safe_prices_df.index)
    safe_prices_df["mid_price"] = pd.to_numeric(col, errors="coerce")

    col = safe_prices_df.get("bid_price_1")
    if col is None:
        col = pd.Series([None] * len(safe_prices_df), index=safe_prices_df.index)
    safe_prices_df["bid_price_1"] = pd.to_numeric(col, errors="coerce")

    col = safe_prices_df.get("ask_price_1")
    if col is None:
        col = pd.Series([None] * len(safe_prices_df), index=safe_prices_df.index)
    safe_prices_df["ask_price_1"] = pd.to_numeric(col, errors="coerce")
    safe_prices_df["spread"] = (
        safe_prices_df["ask_price_1"] - safe_prices_df["bid_price_1"]
    )

    safe_trades_df = trades_df.copy()
    if not safe_trades_df.empty:
        col = safe_trades_df.get("price")
        if col is None:
            col = pd.Series([None] * len(safe_trades_df), index=safe_trades_df.index)
        safe_trades_df["price"] = pd.to_numeric(col, errors="coerce")

        col = safe_trades_df.get("quantity")
        if col is None:
            col = pd.Series([None] * len(safe_trades_df), index=safe_trades_df.index)
        safe_trades_df["quantity"] = pd.to_numeric(col, errors="coerce")
        safe_trades_df["abs_quantity"] = safe_trades_df["quantity"].abs()
        safe_trades_df["notional"] = (
            safe_trades_df["price"] * safe_trades_df["abs_quantity"]
        )

    for product, product_prices in safe_prices_df.groupby("product"):
        mid = product_prices["mid_price"].dropna()
        spread = product_prices["spread"].dropna()

        summary: Dict[str, float | int | None] = {
            "ticks": int(len(product_prices)),
            "mid_min": float(mid.min()) if not mid.empty else None,
            "mid_max": float(mid.max()) if not mid.empty else None,
            "mid_mean": float(mid.mean()) if not mid.empty else None,
            "mid_std": float(mid.std()) if len(mid) > 1 else 0.0,
            "spread_mean": float(spread.mean()) if not spread.empty else None,
            "spread_std": float(spread.std()) if len(spread) > 1 else 0.0,
        }

        trade_symbol_col = "symbol" if "symbol" in safe_trades_df.columns else None
        if trade_symbol_col is not None:
            product_trades = safe_trades_df[safe_trades_df[trade_symbol_col] == product]
        else:
            product_trades = safe_trades_df.iloc[0:0]

        if product_trades.empty:
            summary["trade_count"] = 0
            summary["trade_volume"] = 0
            summary["trade_vwap"] = None
        else:
            total_volume = float(product_trades["abs_quantity"].sum())
            total_notional = float(product_trades["notional"].sum())
            summary["trade_count"] = int(len(product_trades))
            summary["trade_volume"] = int(total_volume)
            summary["trade_vwap"] = (
                float(total_notional / total_volume) if total_volume > 0 else None
            )

        summaries[str(product)] = summary

    return summaries


def plot_analysis_day(
    round_id: int,
    day: int,
    prices_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_prices_df = prices_df.copy()
    col = safe_prices_df.get("timestamp")
    if col is None:
        col = pd.Series([None] * len(safe_prices_df), index=safe_prices_df.index)
    safe_prices_df["timestamp"] = pd.to_numeric(col, errors="coerce")

    col = safe_prices_df.get("mid_price")
    if col is None:
        col = pd.Series([None] * len(safe_prices_df), index=safe_prices_df.index)
    safe_prices_df["mid_price"] = pd.to_numeric(col, errors="coerce")

    safe_trades_df = trades_df.copy()
    if not safe_trades_df.empty:
        col = safe_trades_df.get("timestamp")
        if col is None:
            col = pd.Series([None] * len(safe_trades_df), index=safe_trades_df.index)
        safe_trades_df["timestamp"] = pd.to_numeric(col, errors="coerce")

        col = safe_trades_df.get("quantity")
        if col is None:
            col = pd.Series([None] * len(safe_trades_df), index=safe_trades_df.index)
        safe_trades_df["quantity"] = pd.to_numeric(col, errors="coerce")
        safe_trades_df["abs_quantity"] = safe_trades_df["quantity"].abs()

    plot_module = cast(Any, plt)
    fig, axes = plot_module.subplots(2, 1, figsize=(14, 8), sharex=True)

    for product, group in safe_prices_df.groupby("product"):
        ordered = group.sort_values("timestamp")
        axes[0].plot(
            ordered["timestamp"],
            ordered["mid_price"],
            linewidth=1.2,
            label=str(product),
        )
    axes[0].set_ylabel("Mid Price")
    axes[0].set_title(f"Round {round_id} Day {day}: Mid Price by Product")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    if safe_trades_df.empty:
        axes[1].text(
            0.5,
            0.5,
            "No trade rows available",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_ylabel("Trade Volume")
        axes[1].set_xlabel("Timestamp")
    else:
        trade_volume_ts = (
            safe_trades_df.groupby("timestamp")["abs_quantity"].sum().sort_index()
        )
        axes[1].plot(
            trade_volume_ts.index,
            trade_volume_ts.values,
            color="#1f77b4",
            linewidth=1.2,
        )
        axes[1].set_ylabel("Trade Volume")
        axes[1].set_xlabel("Timestamp")
        axes[1].set_title("Total Trade Volume by Timestamp")
        axes[1].grid(alpha=0.25)

    fig.tight_layout()
    out_path = output_dir / f"round_{round_id}_day_{day}_analysis.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return out_path


def run_data_analysis(
    round_id: int, data_root: Path, output_dir: Path
) -> Dict[str, Any]:
    round_days = discover_round_days(round_id=round_id, data_root=data_root)
    if not round_days:
        raise FileNotFoundError(f"No day files discovered for round {round_id}")

    output_dir.mkdir(parents=True, exist_ok=True)

    day_summaries: List[Dict[str, Any]] = []
    plot_paths: Dict[str, str] = {}

    for day_files in round_days:
        prices_df = pd.read_csv(day_files.prices_path, sep=";")
        trades_df = pd.read_csv(day_files.trades_path, sep=";")

        product_summary = summarize_product_day(
            prices_df=prices_df, trades_df=trades_df
        )
        day_plot = plot_analysis_day(
            round_id=round_id,
            day=day_files.day,
            prices_df=prices_df,
            trades_df=trades_df,
            output_dir=output_dir,
        )

        day_summary: Dict[str, Any] = {
            "day": day_files.day,
            "pricesPath": str(day_files.prices_path),
            "tradesPath": str(day_files.trades_path),
            "rowCount": {
                "prices": int(len(prices_df)),
                "trades": int(len(trades_df)),
            },
            "products": product_summary,
            "plot": str(day_plot),
        }
        day_summaries.append(day_summary)
        plot_paths[str(day_files.day)] = str(day_plot)

    all_products = sorted(
        {
            product
            for day_summary in day_summaries
            for product in day_summary["products"].keys()
        }
    )

    analysis_artifact: Dict[str, Any] = {
        "round": round_id,
        "generatedAtEpochMs": int(time.time() * 1000),
        "dataRoot": str(data_root.resolve()),
        "outputDir": str(output_dir.resolve()),
        "days": day_summaries,
        "products": all_products,
        "plots": plot_paths,
    }

    summary_path = output_dir / f"round_{round_id}_analysis.json"
    summary_path.write_text(json.dumps(analysis_artifact, indent=2), encoding="utf-8")
    analysis_artifact["summaryPath"] = str(summary_path)

    return analysis_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Platform-like local runner: strategy file -> run artifact JSON"
    )
    parser.add_argument("-r", "--round", type=int, required=True, help="Round number")
    parser.add_argument(
        "-s",
        "--strategy",
        type=Path,
        help="Path to strategy submission file (defaults to strategy/roundX/main.py)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root folder containing round directories",
    )
    parser.add_argument(
        "-d",
        "--data-file",
        type=Path,
        help="Specific data file to run instead of scanning data root",
    )
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Save plots for each day"
    )
    parser.add_argument(
        "-a",
        "--analyze",
        action="store_true",
        dest="analyze",
        help="Analyze round market data from data/roundX and save outputs in plots/data",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Optional run id override (default: generated numeric id)",
    )
    return parser


def resolve_strategy_path(round_id: int, cli_path: Path | None) -> Path:
    if cli_path is not None:
        return cli_path.resolve()
    return (Path("strategy") / f"round{round_id}" / "main.py").resolve()


def main() -> int:
    try:
        parser = build_parser()
        args = parser.parse_args()

        if args.analyze:
            analysis_output_dir = plots_dir / "data"
            analysis_artifact = run_data_analysis(
                round_id=args.round,
                data_root=args.data_root,
                output_dir=analysis_output_dir,
            )

            print("Round data analysis complete")
            print(f"Round: {args.round}")
            print(f"Data root: {analysis_artifact['dataRoot']}")
            print(f"Output dir: {analysis_artifact['outputDir']}")
            print(f"Summary JSON: {analysis_artifact['summaryPath']}")
            print("Day plots:")
            for day, path in sorted(analysis_artifact["plots"].items()):
                print(f"  day {day} -> {path}")
            return 0

        strategy_path = resolve_strategy_path(args.round, args.strategy)
        trader_cls = load_trader_class(strategy_path)

        if args.data_file:
            import re

            m = re.search(r"day_(-?\d+)", args.data_file.name)
            days = [int(m.group(1))] if m else [0]
        else:
            round_days = discover_round_days(
                round_id=args.round, data_root=args.data_root
            )
            if not round_days:
                raise FileNotFoundError(
                    f"No day files discovered for round {args.round}"
                )
            days = [item.day for item in round_days]

        run_id = args.run_id or generate_run_id()

        print("Local platform-style replay")
        print(f"Round: {args.round}")
        print(f"Strategy: {strategy_path}")
        if args.data_file:
            print(f"Data File: {args.data_file}")
        else:
            print(f"Days: {', '.join(str(day) for day in days)}")
        print("=" * 72)

        day_results: List[Dict[str, Any]] = []
        plots_by_day: Dict[int, str] = {}

        for day in days:
            frames = build_replay_frames(
                round_id=args.round,
                day=day,
                data_root=args.data_root,
                data_file=args.data_file,
            )
            day_result = run_day(
                day=day,
                frames=frames,
                trader_cls=trader_cls,
            )
            day_results.append(day_result)

            print(
                f"day {day}: pnl={day_result['pnl']:.2f} cash={day_result['cash']:.2f} "
                f"pos={day_result['position']} fills={day_result['fills']} volume={day_result['volume']} "
                f"forced_flatten={day_result['forced_flatten_pnl']:.2f}"
            )

            if args.plot:
                label = f"run_{run_id}_round_{args.round}_day_{day}"
                plot_path = plot_day_result(day_result, plots_dir, label)
                plots_by_day[day] = str(plot_path)
                print(f"  plot -> {plot_path}")

        total_profit = float(sum(day_result["pnl"] for day_result in day_results))

        artifact: Dict[str, Any] = {
            "runId": run_id,
            "round": str(args.round),
            "status": "FINISHED",
            "strategyFile": str(strategy_path),
            "profit": total_profit,
            "dayResults": day_results,
            "plots": {str(day): path for day, path in sorted(plots_by_day.items())},
            "createdAtEpochMs": int(time.time() * 1000),
        }

        json_out = logs_dir / f"{run_id}.json"
        py_out = logs_dir / f"{run_id}.py"

        json_out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        shutil.copyfile(strategy_path, py_out)

        print("-" * 72)
        print(f"Combined profit: {total_profit:.2f}")
        print(f"Artifact JSON -> {json_out}")
        print(f"Artifact strategy copy -> {py_out}")

        return 0
    except RunnerValidationError as exc:
        print(exc)
        return 1
    except (FileNotFoundError, ImportError, AttributeError, ValueError) as exc:
        print(f"Runner error: {exc}")
        return 1
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
