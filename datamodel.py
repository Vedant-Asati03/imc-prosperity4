from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from json import JSONEncoder
from pathlib import Path
from typing import Dict, Iterable, List

import jsonpickle  # type: ignore

Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int

data_dir: Path = Path(__file__).resolve().parents[1] / "data"


class Listing:
    """
    Represents a listing of a product on the exchange.

    Attributes:
        symbol: The symbol of the listing.
        product: The product being listed.
        denomination: The denomination of the listing.
    """

    def __init__(self, symbol: Symbol, product: Product, denomination: Product):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination


class ConversionObservation:
    def __init__(
        self,
        bidPrice: float,
        askPrice: float,
        transportFees: float,
        exportTariff: float,
        importTariff: float,
        # sunlight: float,
        # humidity: float,
    ):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        # self.sugarPrice = sugarPrice
        # self.sunlightIndex = sunlightIndex


class Observation:
    def __init__(
        self,
        plainValueObservations: Dict[Product, ObservationValue],
        conversionObservations: Dict[Product, ConversionObservation],
    ) -> None:
        self.plainValueObservations = plainValueObservations
        self.conversionObservations = conversionObservations

    def __str__(self) -> str:
        return f"(plainValueObservations: {jsonpickle.encode(self.plainValueObservations)}, conversionObservations: {jsonpickle.encode(self.conversionObservations)})"  # type: ignore


class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return f"({self.symbol}, {self.price}, {self.quantity})"

    def __repr__(self) -> str:
        return f"({self.symbol}, {self.price}, {self.quantity})"


class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class Trade:
    def __init__(
        self,
        symbol: Symbol,
        price: int,
        quantity: int,
        buyer: UserId = "",
        seller: UserId = "",
        timestamp: int = 0,
    ) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

    def __str__(self) -> str:
        return f"({self.symbol}, {self.buyer} << {self.seller}, {self.price}, {self.quantity}, {self.timestamp})"

    def __repr__(self) -> str:
        return f"({self.symbol}, {self.buyer} << {self.seller}, {self.price}, {self.quantity}, {self.timestamp})"


class TradingState(object):
    def __init__(
        self,
        traderData: str,
        timestamp: Time,
        listings: Dict[Symbol, Listing],
        order_depths: Dict[Symbol, OrderDepth],
        own_trades: Dict[Symbol, List[Trade]],
        market_trades: Dict[Symbol, List[Trade]],
        position: Dict[Product, Position],
        observations: Observation,
    ):

        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)


class ProsperityEncoder(JSONEncoder):
    def default(self, o: object) -> object:
        return o.__dict__


@dataclass(frozen=True)
class RoundDayFiles:
    day: int
    prices_path: Path
    trades_path: Path


@dataclass(frozen=True)
class ReplayFrame:
    day: int
    timestamp: int
    state: TradingState
    mid_prices: Dict[Product, float]
    profit_and_loss: Dict[Product, float]


def _resolve_data_root(data_root: Path | None = None) -> Path:
    return (data_root or data_dir).resolve()


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    return int(float(stripped))


def _parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    stripped = value.strip()
    if stripped == "":
        return default
    return float(stripped)


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def _extract_day(path: Path, prefix: str) -> int | None:
    pattern = rf"{prefix}_round_\d+_day_(-?\d+)\.csv"
    match = re.fullmatch(pattern, path.name)
    if not match:
        return None
    return int(match.group(1))


def discover_round_days(
    round_id: int,
    data_root: Path | None = None,
    require_trades: bool = True,
) -> List[RoundDayFiles]:
    root = _resolve_data_root(data_root)
    round_dir = root / f"round{round_id}"
    if not round_dir.exists():
        raise FileNotFoundError(f"Round directory does not exist: {round_dir}")

    price_files = sorted(round_dir.glob("prices_round_*_day_*.csv"))
    trade_files = sorted(round_dir.glob("trades_round_*_day_*.csv"))

    price_by_day: Dict[int, Path] = {}
    for path in price_files:
        day = _extract_day(path, "prices")
        if day is not None:
            price_by_day[day] = path.resolve()

    trade_by_day: Dict[int, Path] = {}
    for path in trade_files:
        day = _extract_day(path, "trades")
        if day is not None:
            trade_by_day[day] = path.resolve()

    if not price_by_day:
        raise FileNotFoundError(f"No prices files found in {round_dir}")

    if require_trades:
        missing_trade_days = sorted(
            day for day in price_by_day if day not in trade_by_day
        )
        if missing_trade_days:
            days = ", ".join(str(day) for day in missing_trade_days)
            raise FileNotFoundError(
                f"Missing trades files for round {round_id}, day(s): {days}"
            )

    days = sorted(price_by_day.keys())
    round_files: List[RoundDayFiles] = []
    for day in days:
        trades_path = trade_by_day.get(day)
        if trades_path is None:
            continue
        round_files.append(
            RoundDayFiles(
                day=day,
                prices_path=price_by_day[day],
                trades_path=trades_path,
            )
        )

    return round_files


def discover_products(price_rows: Iterable[Dict[str, str]]) -> List[Product]:
    products = {
        row.get("product", "").strip()
        for row in price_rows
        if row.get("product", "").strip()
    }
    return sorted(products)


def load_day_rows(
    round_id: int,
    day: int,
    data_root: Path | None = None,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    all_days = discover_round_days(round_id=round_id, data_root=data_root)
    day_files = next((item for item in all_days if item.day == day), None)
    if day_files is None:
        available_days = ", ".join(str(item.day) for item in all_days)
        raise FileNotFoundError(
            f"No files found for round {round_id}, day {day}. Available days: {available_days}"
        )

    price_rows = _read_csv_rows(day_files.prices_path)
    trade_rows = _read_csv_rows(day_files.trades_path)
    return price_rows, trade_rows


def build_listings(
    products: Iterable[Product], denomination: Product = "XIRECS"
) -> Dict[Symbol, Listing]:
    return {
        product: Listing(symbol=product, product=product, denomination=denomination)
        for product in sorted(set(products))
    }


def build_order_depth_from_price_row(row: Dict[str, str]) -> OrderDepth:
    depth = OrderDepth()
    for level in (1, 2, 3):
        bid_price = _parse_int(row.get(f"bid_price_{level}"))
        bid_volume = _parse_int(row.get(f"bid_volume_{level}"))
        if bid_price is not None and bid_volume is not None and bid_volume > 0:
            depth.buy_orders[bid_price] = bid_volume

        ask_price = _parse_int(row.get(f"ask_price_{level}"))
        ask_volume = _parse_int(row.get(f"ask_volume_{level}"))
        if ask_price is not None and ask_volume is not None and ask_volume > 0:
            depth.sell_orders[ask_price] = -abs(ask_volume)

    return depth


def build_market_trades_by_timestamp(
    trade_rows: Iterable[Dict[str, str]],
) -> Dict[int, Dict[Symbol, List[Trade]]]:
    grouped: Dict[int, Dict[Symbol, List[Trade]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in trade_rows:
        timestamp = _parse_int(row.get("timestamp"))
        symbol = row.get("symbol", "").strip()
        price = _parse_int(row.get("price"))
        quantity = _parse_int(row.get("quantity"))

        if timestamp is None or not symbol or price is None or quantity is None:
            continue

        grouped[timestamp][symbol].append(
            Trade(
                symbol=symbol,
                price=price,
                quantity=quantity,
                buyer=row.get("buyer", "").strip(),
                seller=row.get("seller", "").strip(),
                timestamp=timestamp,
            )
        )

    result: Dict[int, Dict[Symbol, List[Trade]]] = {}
    for ts in sorted(grouped.keys()):
        result[ts] = {symbol: trades for symbol, trades in grouped[ts].items()}
    return result


def build_replay_frames(
    round_id: int,
    day: int,
    data_root: Path | None = None,
    trader_data: str = "",
    initial_position: Dict[Product, Position] | None = None,
    data_file: Path | None = None,
) -> List[ReplayFrame]:
    if data_file is not None:
        price_rows = _read_csv_rows(data_file)
        trade_rows = []
    else:
        price_rows, trade_rows = load_day_rows(
            round_id=round_id, day=day, data_root=data_root
        )

    products = discover_products(price_rows)
    listings = build_listings(products)
    initial_position = initial_position or {}
    base_position = {
        product: int(initial_position.get(product, 0)) for product in products
    }

    price_rows_by_timestamp: Dict[int, Dict[Product, Dict[str, str]]] = defaultdict(
        dict
    )
    for row in price_rows:
        timestamp = _parse_int(row.get("timestamp"))
        product = row.get("product", "").strip()
        if timestamp is None or not product:
            continue
        price_rows_by_timestamp[timestamp][product] = row

    market_trades_by_timestamp = build_market_trades_by_timestamp(trade_rows)

    frames: List[ReplayFrame] = []
    for timestamp in sorted(price_rows_by_timestamp.keys()):
        rows_by_product = price_rows_by_timestamp[timestamp]
        order_depths = {
            product: build_order_depth_from_price_row(row)
            for product, row in rows_by_product.items()
        }

        mid_prices = {
            product: _parse_float(row.get("mid_price"), 0.0)
            for product, row in rows_by_product.items()
        }
        profit_and_loss = {
            product: _parse_float(row.get("profit_and_loss"), 0.0)
            for product, row in rows_by_product.items()
        }

        market_trades_for_ts = market_trades_by_timestamp.get(timestamp, {})
        own_trades: Dict[Symbol, List[Trade]] = {product: [] for product in products}
        market_trades: Dict[Symbol, List[Trade]] = {
            product: list(market_trades_for_ts.get(product, [])) for product in products
        }

        state = TradingState(
            traderData=trader_data,
            timestamp=timestamp,
            listings=listings,
            order_depths=order_depths,
            own_trades=own_trades,
            market_trades=market_trades,
            position=base_position.copy(),
            observations=Observation(
                plainValueObservations={},
                conversionObservations={},
            ),
        )

        frames.append(
            ReplayFrame(
                day=day,
                timestamp=timestamp,
                state=state,
                mid_prices=mid_prices,
                profit_and_loss=profit_and_loss,
            )
        )

    return frames


def load_round_replay(
    round_id: int,
    data_root: Path | None = None,
    trader_data: str = "",
    initial_position: Dict[Product, Position] | None = None,
) -> Dict[int, List[ReplayFrame]]:
    replay: Dict[int, List[ReplayFrame]] = {}
    for day_files in discover_round_days(round_id=round_id, data_root=data_root):
        replay[day_files.day] = build_replay_frames(
            round_id=round_id,
            day=day_files.day,
            data_root=data_root,
            trader_data=trader_data,
            initial_position=initial_position,
        )
    return replay
