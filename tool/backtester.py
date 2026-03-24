from __future__ import annotations

import argparse
import csv
import importlib.util
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import matplotlib.pyplot as plt

CSVRow = Dict[str, str]
MAX_TRADER_DATA_LEN = 50_000
MAX_RUN_TIME_MS = 900


@dataclass
class OrderDepth:
    buy_orders: Dict[int, int]
    sell_orders: Dict[int, int]


@dataclass
class TradingState:
    traderData: str
    timestamp: int
    order_depths: Dict[str, OrderDepth]
    position: Dict[str, int]


@dataclass
class FillEvent:
    timestamp: int
    product: str
    side: str
    price: int
    quantity: int


@dataclass
class BacktestIssue:
    severity: str
    message: str
    timestamp: int | None = None


class BacktestValidationError(Exception):
    def __init__(self, issues: List[BacktestIssue]) -> None:
        self.issues = issues
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = ["Backtester validation failed:"]
        for issue in self.issues:
            ts = f" [ts={issue.timestamp}]" if issue.timestamp is not None else ""
            lines.append(f"- {issue.severity.upper()}{ts}: {issue.message}")
        return "\n".join(lines)


def has_errors(issues: List[BacktestIssue]) -> bool:
    return any(issue.severity == "error" for issue in issues)


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    return int(float(value))


def discover_price_files(data_root: Path, round_id: int) -> List[Path]:
    round_dir = data_root / f"round{round_id}"
    files: List[Path] = sorted(round_dir.glob("prices_round_*_day_*.csv"))

    files = [path.resolve() for path in files]
    if not files:
        raise FileNotFoundError(f"No price files found in {round_dir}")
    return files


def discover_products(price_files: List[Path]) -> List[str]:
    products: set[str] = set()
    for file_path in price_files:
        with file_path.open("r", newline="") as file:
            reader = csv.DictReader(file, delimiter=";")
            for row in reader:
                product = row.get("product")
                if product:
                    products.add(product)
    return sorted(products)


def load_trader_class(round_id: int) -> type[Any]:
    strategy_path = Path("strategy") / f"round{round_id}" / "main.py"
    strategy_path = strategy_path.resolve()

    if not strategy_path.exists():
        raise FileNotFoundError(
            f"Strategy file not found: {strategy_path}. Expected strategy/round{round_id}/main.py"
        )

    module_name = f"strategy_round_{round_id}_main"
    spec = importlib.util.spec_from_file_location(module_name, strategy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load strategy module from {strategy_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    trader_cls = getattr(module, "Trader", None)
    if trader_cls is None:
        raise AttributeError(
            f"Strategy file {strategy_path} does not define Trader class"
        )

    return trader_cls


def load_price_rows(file_path: Path) -> List[CSVRow]:
    rows: List[CSVRow] = []
    with file_path.open("r", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def build_snapshots(rows: List[CSVRow]) -> Dict[int, Dict[str, OrderDepth]]:
    snapshots: Dict[int, Dict[str, OrderDepth]] = defaultdict(dict)

    for row in rows:
        timestamp = int(row["timestamp"])
        product = row["product"]

        bids: Dict[int, int] = {}
        asks: Dict[int, int] = {}

        for level in (1, 2, 3):
            bid_price = parse_int(row.get(f"bid_price_{level}", ""))
            bid_vol = parse_int(row.get(f"bid_volume_{level}", ""))
            if bid_price is not None and bid_vol is not None and bid_vol > 0:
                bids[bid_price] = bid_vol

            ask_price = parse_int(row.get(f"ask_price_{level}", ""))
            ask_vol = parse_int(row.get(f"ask_volume_{level}", ""))
            if ask_price is not None and ask_vol is not None and ask_vol > 0:
                asks[ask_price] = -ask_vol

        snapshots[timestamp][product] = OrderDepth(buy_orders=bids, sell_orders=asks)

    return dict(sorted(snapshots.items()))


def load_mid_prices(rows: List[CSVRow]) -> Dict[int, Dict[str, float]]:
    mids: Dict[int, Dict[str, float]] = defaultdict(dict)
    for row in rows:
        timestamp = int(row["timestamp"])
        product = row["product"]
        mids[timestamp][product] = float(row["mid_price"])
    return dict(sorted(mids.items()))


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
) -> Tuple[float, int, int, List[FillEvent], List[str]]:
    fills = 0
    volume = 0
    fill_events: List[FillEvent] = []
    rejected_products: List[str] = []

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
            qty_remaining = order.quantity

            if qty_remaining > 0:
                for ask_price in sorted(depth.sell_orders.keys()):
                    if qty_remaining <= 0:
                        break
                    if order.price < ask_price:
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

            elif qty_remaining < 0:
                qty_to_sell = -qty_remaining
                for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                    if qty_to_sell <= 0:
                        break
                    if order.price > bid_price:
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

    return cash, fills, volume, fill_events, rejected_products


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


def run_backtest(
    prices_csv: Path,
    trader_cls: type[Any],
    verbose: bool = False,
    verbose_every: int = 1,
    verbose_only_active: bool = False,
) -> Dict[str, Any]:
    rows = load_price_rows(prices_csv)
    snapshots = build_snapshots(rows)
    mids = load_mid_prices(rows)

    trader = trader_cls()
    trader_data = ""

    effective_limits: Dict[str, int] = dict(getattr(trader, "POSITION_LIMITS", {}))

    all_products = sorted({row["product"] for row in rows})

    missing_limits = [
        product for product in all_products if product not in effective_limits
    ]
    if missing_limits:
        raise BacktestValidationError(
            [
                BacktestIssue(
                    severity="error",
                    message=(
                        "Missing POSITION_LIMITS entries for product(s): "
                        f"{', '.join(missing_limits)}"
                    ),
                )
            ]
        )

    position: Dict[str, int] = {product: 0 for product in all_products}

    cash = 0.0
    total_fills = 0
    total_volume = 0
    warnings_log: List[BacktestIssue] = []
    timeline: Dict[str, Any] = {
        "timestamp": [],
        "cash": [],
        "pnl": [],
        "mid_by_product": {product: [] for product in all_products},
        "pos_by_product": {product: [] for product in all_products},
    }

    for timestamp, snapshot in snapshots.items():
        state = TradingState(
            traderData=trader_data,
            timestamp=timestamp,
            order_depths=snapshot,
            position=position.copy(),
        )

        issues: List[BacktestIssue] = []

        orders_by_product: Dict[str, Any] = {}
        conversions: Any = 0
        trader_data_out: Any = ""

        start = time.perf_counter()
        try:
            raw_result_obj: Any = trader.run(state)
        except Exception as exc:
            raise BacktestValidationError(
                [
                    BacktestIssue(
                        severity="error",
                        timestamp=timestamp,
                        message=f"Trader.run raised {type(exc).__name__}: {exc}",
                    )
                ]
            ) from exc

        if not isinstance(raw_result_obj, tuple):
            issues.append(
                BacktestIssue(
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
                    BacktestIssue(
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
                        BacktestIssue(
                            severity="error",
                            timestamp=timestamp,
                            message="orders_by_product must be a dict[str, list[Order]]",
                        )
                    )

        trader_data = trader_data_out

        for product_key, product_orders in orders_by_product.items():
            product = str(product_key)
            if product not in snapshot:
                issues.append(
                    BacktestIssue(
                        severity="error",
                        timestamp=timestamp,
                        message=(
                            f"Trader returned orders for unknown product '{product}' "
                            "at this timestamp"
                        ),
                    )
                )
                continue

            if not isinstance(product_orders, list):
                issues.append(
                    BacktestIssue(
                        severity="error",
                        timestamp=timestamp,
                        message=f"Orders for product '{product}' must be a list",
                    )
                )
                continue

            orders_list = cast(List[Any], product_orders)
            for order in orders_list:
                for field in ("symbol", "price", "quantity"):
                    if not hasattr(order, field):
                        issues.append(
                            BacktestIssue(
                                severity="error",
                                timestamp=timestamp,
                                message=(
                                    f"Malformed order for '{product}': missing field '{field}'"
                                ),
                            )
                        )
                        break
                else:
                    symbol = str(getattr(order, "symbol"))
                    if symbol != product:
                        issues.append(
                            BacktestIssue(
                                severity="error",
                                timestamp=timestamp,
                                message=(
                                    "Order symbol mismatch: "
                                    f"dict key '{product}' but order.symbol='{symbol}'"
                                ),
                            )
                        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if elapsed_ms > MAX_RUN_TIME_MS:
            issues.append(
                BacktestIssue(
                    severity="error",
                    timestamp=timestamp,
                    message=(
                        f"Trader.run exceeded {MAX_RUN_TIME_MS}ms: {elapsed_ms:.2f}ms"
                    ),
                )
            )

        if not isinstance(trader_data, str):
            issues.append(
                BacktestIssue(
                    severity="warning",
                    timestamp=timestamp,
                    message=(
                        "traderData is not a string; coercing to string for compatibility"
                    ),
                )
            )
            trader_data = str(trader_data)

        if len(trader_data) > MAX_TRADER_DATA_LEN:
            issues.append(
                BacktestIssue(
                    severity="warning",
                    timestamp=timestamp,
                    message=(
                        f"traderData length {len(trader_data)} exceeds {MAX_TRADER_DATA_LEN}; truncating"
                    ),
                )
            )
            trader_data = trader_data[:MAX_TRADER_DATA_LEN]

        if conversions not in (0, None):
            issues.append(
                BacktestIssue(
                    severity="error",
                    timestamp=timestamp,
                    message=(
                        "This backtester does not simulate conversions. "
                        "Trader must return conversions=0 or None."
                    ),
                )
            )

        warnings_log.extend(issue for issue in issues if issue.severity == "warning")
        if has_errors(issues):
            raise BacktestValidationError(issues)

        cash, fills, volume, fill_events, rejected_products = execute_orders(
            timestamp=timestamp,
            orders_by_product=cast(Dict[str, List[Any]], orders_by_product),
            snapshot=snapshot,
            position=position,
            cash=cash,
            limits=effective_limits,
        )
        total_fills += fills
        total_volume += volume

        pnl_now = mark_to_market(cash, position, mids[timestamp])
        timeline["timestamp"].append(timestamp)
        timeline["cash"].append(cash)
        timeline["pnl"].append(pnl_now)
        for product in all_products:
            timeline["mid_by_product"][product].append(
                mids[timestamp].get(product, 0.0)
            )
            timeline["pos_by_product"][product].append(position.get(product, 0))

        if verbose and timestamp % max(1, verbose_every) == 0:
            order_desc: List[str] = []
            for product_orders in orders_by_product.values():
                if not product_orders:
                    continue
                formatted = ", ".join(
                    f"{order.symbol}:{'B' if order.quantity > 0 else 'S'}{abs(order.quantity)}@{order.price}"
                    for order in product_orders
                )
                order_desc.append(formatted)

            has_activity = bool(order_desc or fill_events or rejected_products)
            if verbose_only_active and not has_activity:
                continue

            print(
                f"ts={timestamp} orders=[{' | '.join(order_desc) if order_desc else 'none'}]"
            )

            if rejected_products:
                print(f"  rejected(position-limit): {', '.join(rejected_products)}")

            if fill_events:
                fills_txt = ", ".join(
                    f"{event.product}:{event.side}{event.quantity}@{event.price}"
                    for event in fill_events
                )
                print(f"  fills: {fills_txt}")
            else:
                print("  fills: none")

            mid_state = {
                product: mids[timestamp].get(product, 0.0) for product in all_products
            }
            print(
                "  state: "
                f"pos={position} cash={cash:.2f} pnl={pnl_now:.2f} "
                f"mid={mid_state}"
            )

    final_ts = max(mids.keys())
    final_pnl = mark_to_market(cash, position, mids[final_ts])
    flatten_pnl = cash + forced_flatten_value(position, snapshots[final_ts])

    return {
        "file": prices_csv.name,
        "path": str(prices_csv),
        "products": all_products,
        "final_pnl": final_pnl,
        "cash": cash,
        "position": dict(position),
        "fills": total_fills,
        "volume": total_volume,
        "forced_flatten_pnl": flatten_pnl,
        "timeline": timeline,
        "warnings": warnings_log,
    }


def plot_result(result: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline = result["timeline"]
    ts = timeline["timestamp"]
    products: List[str] = result["products"]

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
    axes[0].set_title(result["file"])
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
    output_path = output_dir / f"{Path(result['file']).stem}_strategy.png"
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generic Prosperity backtester (crossing fills only)"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root folder containing round directories",
    )
    parser.add_argument(
        "--round",
        type=int,
        required=True,
        help="Round number, e.g. --round 0",
    )
    parser.add_argument("--plot", action="store_true", help="Save strategy plots")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where plots are written when --plot is used",
    )
    parser.add_argument(
        "--verbosity",
        choices=["off", "active", "all"],
        default="off",
        help="Verbosity mode: off|active|all",
    )
    parser.add_argument(
        "--verbosity-step",
        type=int,
        default=1,
        help="Print every Nth timestamp when verbosity is active/all",
    )
    return parser


def main() -> int:
    try:
        parser = build_parser()
        args = parser.parse_args()

        price_files = discover_price_files(
            data_root=args.data_root, round_id=args.round
        )
        available_products = discover_products(price_files)
        trader_cls = load_trader_class(args.round)

        verbose = args.verbosity in {"active", "all"}
        verbose_only_active = args.verbosity == "active"

        print("Local replay (crossing fills only)")
        print(f"Round: {args.round}")
        print(f"Files: {len(price_files)}")
        print(f"Products: {', '.join(available_products)}")
        print("=" * 72)

        all_results: List[Dict[str, Any]] = []
        for path in price_files:
            result = run_backtest(
                prices_csv=path,
                trader_cls=trader_cls,
                verbose=verbose,
                verbose_every=args.verbosity_step,
                verbose_only_active=verbose_only_active,
            )
            all_results.append(result)

        combined_pnl: float = 0.0
        for result in all_results:
            combined_pnl += result["final_pnl"]
            print(
                f"{result['file']}: pnl={result['final_pnl']:.2f} cash={result['cash']:.2f} "
                f"pos={result['position']} fills={result['fills']} volume={result['volume']} "
                f"forced_flatten={result['forced_flatten_pnl']:.2f}"
            )
            warnings: List[BacktestIssue] = result.get("warnings", [])
            if warnings:
                print(f"  warnings: {len(warnings)}")
                for warning in warnings[:5]:
                    ts = (
                        f" [ts={warning.timestamp}]"
                        if warning.timestamp is not None
                        else ""
                    )
                    print(f"    -{ts} {warning.message}")
                if len(warnings) > 5:
                    print(f"    ... {len(warnings) - 5} more")
            if args.plot:
                plot_path = plot_result(result, args.output_dir)
                print(f"  plot -> {plot_path}")

        print("-" * 72)
        print(f"Combined pnl: {combined_pnl:.2f}")
        return 0
    except BacktestValidationError as exc:
        print(exc)
        return 1
    except (FileNotFoundError, ImportError, AttributeError, ValueError) as exc:
        print(f"Backtester error: {exc}")
        return 1
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
