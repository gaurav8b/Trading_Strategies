# -*- coding: utf-8 -*-
"""
AuctionBot BTC Live / Backtest
==============================

Clean refactor direction:
- auction_bot_qqq.py is the strategy base.
- BTC crypto data connectivity, live price polling, JSON persistence, email alerts,
  and broker-router dispatch are taken from the BTC live bot pattern.
- Legacy BTC static-z / static-AVWAP / layered strategy code is deliberately not
  included, so there is only one active strategy engine in this file.

Strategy retained from the QQQ AuctionBot:
- Rolling VWAP long lower-band breach + uptick trigger.
- Rolling VWAP short upper-band breach + downtick trigger.
- Long-biased paired short logic: shorts require a profitable unlocked long anchor.
- Long and short strategy sizes are separate globals.
- Long inventory locks while its paired short is active.
- Standard target exits plus reset/recovery pair close.
- ATR-scaled entry spacing, adapted for BTC rather than using QQQ point constants.
"""

# =============================================================================
# PORTABLE DEPENDENCY BOOTSTRAP
# =============================================================================

import importlib.util
import subprocess
import sys

_REQUIRED_PACKAGES = {
    "alpaca": "alpaca-py",
    "loguru": "loguru",
    "pytz": "pytz",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
}

for _import_name, _pip_name in _REQUIRED_PACKAGES.items():
    if importlib.util.find_spec(_import_name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pip_name])

# =============================================================================
# IMPORTS
# =============================================================================

import os
import json
import time
import signal
import smtplib
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
from loguru import logger

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestTradeRequest
from alpaca.data.timeframe import TimeFrame

warnings.filterwarnings("ignore")

# =============================================================================
# OPTIONAL BROKER ROUTER IMPORTS
# =============================================================================
# The BTC live bot uses broker_config.py and broker_router.py. This clean file will
# use them when available. If they are not present, the strategy still runs in
# SIM/logging mode without crashing.
# =============================================================================

_export_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _export_path not in sys.path:
    sys.path.insert(0, _export_path)

try:
    from broker_config import BROKER_BACKEND  # type: ignore
except Exception:
    BROKER_BACKEND = "SIM"

try:
    from broker_router import initialize_broker, shutdown_broker, send_broker_order  # type: ignore
except Exception:
    def initialize_broker(logger=None, send_orders_enabled=False):
        if logger:
            logger.info(
                "broker_router.py not found. Running with local SIM broker stub. "
                f"send_orders_enabled={send_orders_enabled}"
            )
        return {"backend": "SIM", "send_orders_enabled": bool(send_orders_enabled)}

    def shutdown_broker(logger=None):
        if logger:
            logger.info("SIM broker stub shutdown complete")
        return {"backend": "SIM", "status": "shutdown"}

    def send_broker_order(env, direction, action, qty, send_orders_enabled, logger=None):
        response = {
            "backend": "SIM",
            "env": env,
            "direction": direction,
            "action": action,
            "qty": qty,
            "send_orders_enabled": bool(send_orders_enabled),
            "sent": False,
            "reason": "broker_router.py unavailable; local SIM stub used",
        }
        if logger:
            logger.info(f"SIM broker order stub | response={response}")
        return response

# =============================================================================
# TIMEZONES
# =============================================================================

tz_utc = timezone.utc
tz_est = pytz.timezone("US/Eastern")

# =============================================================================
# RUN MODE
# =============================================================================
# MODE="LIVE" runs continuously using Alpaca latest crypto trades.
# MODE="BACKTEST" downloads historical BTC minute bars and runs the same strategy.
# =============================================================================

MODE = "LIVE"  # "LIVE" or "BACKTEST"
ENV = "DEMO"

# =============================================================================
# SYMBOL / BROKER CONFIGURATION
# =============================================================================

SYMBOL = "BTC/USD"
BROKER_SYMBOL_LONG = "BTCH6"
BROKER_SYMBOL_SHORT = "BTCM6"

ALPACA_API_KEY = "ALPACA_API_KEY"
ALPACA_SECRET_KEY = "ALPACA_SECRET_KEY"

SEND_BROKER_ORDERS = False

# Broker quantities are execution quantities. Strategy trade sizes below are PnL
# multipliers / unit weights and are preserved separately.
BROKER_ORDER_SIZE_LONG = 1
BROKER_ORDER_SIZE_SHORT = 1

# =============================================================================
# AUCTIONBOT STRATEGY CONFIGURATION — QQQ STRATEGY PORTED TO BTC
# =============================================================================
# Anchor defaults are inherited from the QQQ AuctionBot. They are bar counts, so
# they remain valid for BTC minute bars.
# =============================================================================

START_DATE = "2020-01-01"

LONG_ANCHORS = [4800]
SHORT_ANCHORS = [600]

LONG_ENTRY_STDS = [0.9]
SHORT_ENTRY_STDS = [0.9]

LONG_EXIT_STDS = [0.9]
SHORT_EXIT_STDS = [0.1]

MAX_LONG_POSITIONS = 6
MAX_SHORT_POSITIONS = 3

# Strategy sizing multipliers from the AuctionBot strategy file.
# These are intentionally distinct from broker quantities above.
LONG_TRADE_SIZE = 2.0
SHORT_TRADE_SIZE = 1.0

BIAS_DIRECTION = "LONG"
DIRECTION_GATE = 0

ENTRY_SPACING_MODE = "WEEKLY_ATR"
ENTRY_SPACING_MULT = 0.25
WEEKLY_ATR_LENGTH = 5

# BTC adaptation of the QQQ MIN_ENTRY_DISTANCE concept. The original QQQ file had
# MIN_ENTRY_DISTANCE = 2.5 points. A fixed 2.5 BTC dollars is too small for BTC,
# so the same feature is kept but scaled to the traded instrument.
USE_ADAPTIVE_MIN_ENTRY_DISTANCE = True
MANUAL_MIN_ENTRY_DISTANCE = 2.5
MIN_ENTRY_DISTANCE_PRICE_FRACTION = 0.0005       # 0.05% of BTC price floor
MIN_ENTRY_DISTANCE_STD_FRACTION = 0.10           # 10% of active rolling sigma floor
MIN_ENTRY_DISTANCE_ABSOLUTE_FLOOR = 5.0          # emergency lower floor in BTC dollars

RESET_HORIZON = 10

# Source strategy had RESET_TRIGGER = -10 points. For BTC, default is adaptive.
USE_ADAPTIVE_RESET_TRIGGER = True
MANUAL_RESET_TRIGGER = -10.0
RESET_TRIGGER_PRICE_FRACTION = 0.002             # 0.20% adverse move floor
RESET_TRIGGER_WEEKLY_ATR_FRACTION = 0.05         # 5% of weekly ATR
RESET_TRIGGER_STD_FRACTION = 0.50                # 50% of active short sigma
RESET_TRIGGER_ABSOLUTE_FLOOR = 25.0

# Live engine cadence.
VWAP_RECOMPUTE_INTERVAL_SECONDS = 60
LIVE_LOOP_SLEEP_SECONDS = 5
MARKET_DATA_LOOKBACK_DAYS = 40
HEARTBEAT_FREQUENCY_MINUTES = 2
LOG_DISPLAY_FREQUENCY_MINUTES = 5
PNL_SUMMARY_UPDATE_SECONDS = 60 * 60

RESTRICTED_HOURS_EST: List[int] = []

# Optional live price sanity filter copied from the BTC live bot pattern.
ENABLE_SPURIOUS_CHECK_AGGREGATED = False
ENABLE_SPURIOUS_CHECK_LIVE = False
LIVE_PRICE_BUFFER_SIZE = 5
LIVE_PRICE_OUTLIER_FACTOR = 3.0

# =============================================================================
# EMAIL ALERT CONFIGURATION
# =============================================================================

ENABLE_EMAIL_ALERTS = True
EMAIL_SMTP_SERVER = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587
EMAIL_SENDER = "EMAIL_SENDER"
EMAIL_PASSWORD = "EMAIL_PASSWORD"
EMAIL_RECIPIENTS = [
    "EMAIL_Recipient 1",
    "EMAIL_Recipient 2",
    "EMAIL_Recipient 3",
]

# =============================================================================
# PATHS / LOGGING
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
LOGS_DIR = os.path.join(BASE_DIR, "logs_btc_auctionbot")
os.makedirs(LOGS_DIR, exist_ok=True)

TRADE_LOG_FILE = os.path.join(LOGS_DIR, "trade_log.json")
LIVE_STATUS_FILE = os.path.join(LOGS_DIR, "live_status.json")
CURRENT_PRICE_FILE = os.path.join(LOGS_DIR, "current_price.json")
OPENED_TRADES_FILE = os.path.join(LOGS_DIR, "opened_trades.json")
STATISTICS_FILE = os.path.join(LOGS_DIR, "statistics.json")
LOG_FILE = os.path.join(LOGS_DIR, "trading_bot.log")
PNL_SUMMARY_FILE = os.path.join(LOGS_DIR, "pnl_summary.json")
INTERNALS_FILE = os.path.join(LOGS_DIR, "internals.json")

logger.remove()
logger.add(
    LOG_FILE,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
    rotation="10 MB",
    retention="10 days",
)
logger.add(sys.stderr, level="INFO")

# =============================================================================
# GLOBAL STATE
# =============================================================================

client = CryptoHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

open_positions: List[Dict[str, Any]] = []
last_log_times: Dict[str, datetime] = {}
live_price_buffer: List[float] = []

previous_price: Optional[float] = None
last_pnl_summary_update: Optional[datetime] = None
latest_market_structure: Dict[str, Any] = {}

long_armed = False
short_armed = False
long_arm_extreme: Optional[float] = None
short_arm_extreme: Optional[float] = None

# =============================================================================
# DATA STRUCTURES FOR BACKTEST
# =============================================================================

@dataclass
class Trade:
    side: str
    entry_price: float
    entry_time: pd.Timestamp
    target: float
    size: float = 1.0
    reset_trigger: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    pnl: float = 0.0
    closed: bool = False
    inventory_lock: bool = False
    recovery_owner: Optional[Any] = None
    counter_leg: Optional[Any] = None
    is_hedge_component: bool = False

    def __post_init__(self):
        self.entry_price = float(self.entry_price)
        self.entry_time = pd.Timestamp(self.entry_time)
        self.target = float(self.target)
        self.size = float(self.size)
        if self.size <= 0:
            raise ValueError("Trade size must be positive.")

# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def utc_now() -> datetime:
    return datetime.now(tz_utc)


def est_now_str() -> str:
    return datetime.now(tz_est).strftime("%Y-%m-%d %H:%M:%S")


def format_ts(ts: Any) -> str:
    if ts is None:
        return "N/A"
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def is_restricted_hour_est() -> bool:
    now_est = datetime.now(tz_est)
    return now_est.hour in RESTRICTED_HOURS_EST


def safe_mean(arr: Any) -> float:
    return float(np.mean(arr)) if len(arr) else 0.0


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def _json_default(obj: Any):
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, default=_json_default)
        f.flush()
        os.fsync(f.fileno())


def read_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        if default is not None:
            return default
        if path in (TRADE_LOG_FILE, OPENED_TRADES_FILE):
            return []
        if path == STATISTICS_FILE:
            return init_statistics_template()
        return {}
    except Exception as exc:
        logger.error(f"Read JSON error {path}: {exc}")
        return default


def log_message_throttled(message: str, key: str) -> None:
    now = utc_now()
    last = last_log_times.get(key)
    if last is None or (now - last).total_seconds() > LOG_DISPLAY_FREQUENCY_MINUTES * 60:
        logger.info(message)
        last_log_times[key] = now


def log_heartbeat() -> None:
    logger.info("Heartbeat: AuctionBot BTC live runner active")

# =============================================================================
# STRATEGY SCALING HELPERS
# =============================================================================

def compute_adaptive_min_entry_distance(
    price: float,
    weekly_atr: Optional[float] = None,
    rolling_std: Optional[float] = None,
) -> float:
    """BTC-safe equivalent of the QQQ strategy's MIN_ENTRY_DISTANCE feature."""
    if not USE_ADAPTIVE_MIN_ENTRY_DISTANCE:
        return float(MANUAL_MIN_ENTRY_DISTANCE)

    candidates = [float(MIN_ENTRY_DISTANCE_ABSOLUTE_FLOOR)]

    if price is not None and np.isfinite(price) and price > 0:
        candidates.append(float(price) * float(MIN_ENTRY_DISTANCE_PRICE_FRACTION))

    if weekly_atr is not None and np.isfinite(weekly_atr) and weekly_atr > 0:
        candidates.append(float(weekly_atr) * 0.01)

    if rolling_std is not None and np.isfinite(rolling_std) and rolling_std > 0:
        candidates.append(float(rolling_std) * float(MIN_ENTRY_DISTANCE_STD_FRACTION))

    return float(max(candidates))


def compute_entry_spacing(
    price: float,
    weekly_atr: Optional[float],
    rolling_std: Optional[float],
) -> float:
    floor = compute_adaptive_min_entry_distance(price, weekly_atr, rolling_std)
    if weekly_atr is None or not np.isfinite(weekly_atr) or weekly_atr <= 0:
        return float(floor)
    return float(max(floor, float(weekly_atr) * float(ENTRY_SPACING_MULT)))


def compute_reset_trigger_points(
    price: float,
    weekly_atr: Optional[float] = None,
    rolling_std: Optional[float] = None,
) -> float:
    """Negative raw-PnL threshold used by the pair reset/recovery rule."""
    if not USE_ADAPTIVE_RESET_TRIGGER:
        return float(MANUAL_RESET_TRIGGER)

    candidates = [float(RESET_TRIGGER_ABSOLUTE_FLOOR)]

    if price is not None and np.isfinite(price) and price > 0:
        candidates.append(float(price) * float(RESET_TRIGGER_PRICE_FRACTION))

    if weekly_atr is not None and np.isfinite(weekly_atr) and weekly_atr > 0:
        candidates.append(float(weekly_atr) * float(RESET_TRIGGER_WEEKLY_ATR_FRACTION))

    if rolling_std is not None and np.isfinite(rolling_std) and rolling_std > 0:
        candidates.append(float(rolling_std) * float(RESET_TRIGGER_STD_FRACTION))

    return -float(max(candidates))

# =============================================================================
# ALPACA CRYPTO DATA
# =============================================================================

def filter_spurious_aggregated(prices: np.ndarray, volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not ENABLE_SPURIOUS_CHECK_AGGREGATED or len(prices) < 5:
        return prices, volumes

    mean = float(np.mean(prices))
    std = float(np.std(prices))
    if std == 0:
        return prices, volumes

    mask = np.abs(prices - mean) <= 3 * std
    if int(mask.sum()) < 2:
        return prices, volumes
    return prices[mask], volumes[mask]


def filter_live_price(price: float) -> Optional[float]:
    live_price_buffer.append(float(price))
    if len(live_price_buffer) > LIVE_PRICE_BUFFER_SIZE:
        live_price_buffer.pop(0)

    if not ENABLE_SPURIOUS_CHECK_LIVE or len(live_price_buffer) < 3:
        return float(price)

    mean = float(np.mean(live_price_buffer))
    std = float(np.std(live_price_buffer))
    if std > 0 and abs(float(price) - mean) > LIVE_PRICE_OUTLIER_FACTOR * std:
        logger.warning(f"Ignoring spurious live price: {price}")
        return None
    return float(price)


def _normalize_crypto_bars_dataframe(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = raw_df.reset_index() if isinstance(raw_df.index, pd.MultiIndex) or "timestamp" not in raw_df.columns else raw_df.copy()
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = df["close"]

    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    return df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()


def fetch_data(symbol: str = SYMBOL, start_date: str = START_DATE) -> pd.DataFrame:
    """Backtest data fetch: BTC/USD minute bars from Alpaca crypto data."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=tz_utc)
    end_dt = utc_now()

    request = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=end_dt,
    )

    bars = client.get_crypto_bars(request)
    return _normalize_crypto_bars_dataframe(bars.df, symbol)


def fetch_intraday_dataframe() -> Optional[pd.DataFrame]:
    """Live structure data fetch: rolling BTC minute window."""
    end_utc = utc_now()
    start_utc = end_utc - timedelta(days=MARKET_DATA_LOOKBACK_DAYS)

    request = CryptoBarsRequest(
        symbol_or_symbols=[SYMBOL],
        timeframe=TimeFrame.Minute,
        start=start_utc,
        end=end_utc,
    )

    try:
        bars = client.get_crypto_bars(request)
        df = _normalize_crypto_bars_dataframe(bars.df, SYMBOL)
    except Exception as exc:
        logger.error(f"fetch_intraday_dataframe error: {exc}")
        return None

    if df is None or df.empty:
        return None
    return df


def fetch_latest_price() -> Optional[float]:
    req = CryptoLatestTradeRequest(symbol_or_symbols=[SYMBOL])
    trades = client.get_crypto_latest_trade(req)
    if SYMBOL not in trades:
        return None
    return float(trades[SYMBOL].price)

# =============================================================================
# VWAP / ATR STRUCTURE
# =============================================================================

def add_weekly_atr(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    daily = out.copy()
    daily["date"] = daily["timestamp"].dt.floor("D")

    daily_ohlc = daily.groupby("date").agg(
        high=("close", "max"),
        low=("close", "min"),
        close=("close", "last"),
    )

    weekly = daily_ohlc.resample("W").agg({
        "high": "max",
        "low": "min",
        "close": "last",
    })

    weekly["prev_close"] = weekly["close"].shift(1)
    tr1 = weekly["high"] - weekly["low"]
    tr2 = (weekly["high"] - weekly["prev_close"]).abs()
    tr3 = (weekly["low"] - weekly["prev_close"]).abs()
    weekly["tr"] = np.maximum.reduce([tr1, tr2, tr3])
    weekly["weekly_atr"] = weekly["tr"].rolling(WEEKLY_ATR_LENGTH).mean()

    daily["week"] = daily["timestamp"].dt.to_period("W").astype(str)
    weekly["week"] = weekly.index.to_period("W").astype(str)

    merged = daily.merge(weekly[["week", "weekly_atr"]], on="week", how="left")
    out["weekly_atr"] = merged["weekly_atr"].values
    return out


def prepare_strategy_dataframe(
    df: pd.DataFrame,
    long_anchor: int,
    long_entry_std: float,
    short_anchor: int,
    short_entry_std: float,
) -> pd.DataFrame:
    out = df.copy()
    prices = out["close"].astype(float)
    volumes = out["volume"].astype(float)
    pv = prices * volumes

    long_vol = volumes.rolling(long_anchor).sum()
    short_vol = volumes.rolling(short_anchor).sum()

    out["long_vwap"] = pv.rolling(long_anchor).sum() / long_vol
    out["long_std"] = prices.rolling(long_anchor).std()
    out["long_lower"] = out["long_vwap"] - float(long_entry_std) * out["long_std"]

    out["short_vwap"] = pv.rolling(short_anchor).sum() / short_vol
    out["short_std"] = prices.rolling(short_anchor).std()
    out["short_upper"] = out["short_vwap"] + float(short_entry_std) * out["short_std"]

    out["vwap"] = out["long_vwap"]
    out["std"] = out["long_std"]
    out["upper"] = out["short_upper"]
    out["lower"] = out["long_lower"]

    out = add_weekly_atr(out)

    out["entry_spacing_dynamic"] = [
        compute_entry_spacing(
            price=float(price),
            weekly_atr=None if pd.isna(weekly_atr) else float(weekly_atr),
            rolling_std=None if pd.isna(rolling_std) else float(rolling_std),
        )
        for price, weekly_atr, rolling_std in zip(out["close"], out["weekly_atr"], out["std"])
    ]

    out["reset_trigger_dynamic"] = [
        compute_reset_trigger_points(
            price=float(price),
            weekly_atr=None if pd.isna(weekly_atr) else float(weekly_atr),
            rolling_std=None if pd.isna(rolling_std) else float(rolling_std),
        )
        for price, weekly_atr, rolling_std in zip(out["close"], out["weekly_atr"], out["short_std"])
    ]

    return out


def compute_market_structure_single(
    long_anchor: int,
    long_entry_std: float,
    long_exit_std: float,
    short_anchor: int,
    short_entry_std: float,
    short_exit_std: float,
) -> Optional[Dict[str, Any]]:
    df = fetch_intraday_dataframe()
    if df is None or df.empty:
        return None

    required = max(long_anchor, short_anchor) + 10
    if len(df) < required:
        logger.warning(
            f"Not enough BTC bars for structure {long_anchor}_{short_anchor}: "
            f"have={len(df)} need={required}. Increase MARKET_DATA_LOOKBACK_DAYS."
        )
        return None

    prepared = prepare_strategy_dataframe(df, long_anchor, long_entry_std, short_anchor, short_entry_std)
    latest = prepared.iloc[-1]

    required_cols = [
        "long_vwap", "long_std", "long_lower",
        "short_vwap", "short_std", "short_upper",
        "entry_spacing_dynamic", "reset_trigger_dynamic",
    ]
    if any(pd.isna(latest[col]) for col in required_cols):
        return None

    return {
        "timestamp": str(latest["timestamp"]),
        "current_price": float(latest["close"]),
        "long_anchor": int(long_anchor),
        "short_anchor": int(short_anchor),
        "long_entry_std": float(long_entry_std),
        "long_exit_std": float(long_exit_std),
        "short_entry_std": float(short_entry_std),
        "short_exit_std": float(short_exit_std),
        "long_vwap": float(latest["long_vwap"]),
        "long_std": float(latest["long_std"]),
        "long_lower": float(latest["long_lower"]),
        "short_vwap": float(latest["short_vwap"]),
        "short_std": float(latest["short_std"]),
        "short_upper": float(latest["short_upper"]),
        "weekly_atr": None if pd.isna(latest.get("weekly_atr")) else float(latest["weekly_atr"]),
        "entry_spacing_dynamic": float(latest["entry_spacing_dynamic"]),
        "reset_trigger_dynamic": float(latest["reset_trigger_dynamic"]),
        "signal_source": f"{long_anchor}_{short_anchor}",
    }


def compute_all_market_structures() -> List[Dict[str, Any]]:
    structures: List[Dict[str, Any]] = []
    for long_anchor in LONG_ANCHORS:
        for long_entry in LONG_ENTRY_STDS:
            for long_exit in LONG_EXIT_STDS:
                for short_anchor in SHORT_ANCHORS:
                    for short_entry in SHORT_ENTRY_STDS:
                        for short_exit in SHORT_EXIT_STDS:
                            structure = compute_market_structure_single(
                                long_anchor,
                                long_entry,
                                long_exit,
                                short_anchor,
                                short_entry,
                                short_exit,
                            )
                            if structure is not None:
                                structures.append(structure)

    structures = sorted(structures, key=lambda s: int(s["long_anchor"]), reverse=True)
    return structures


def debug_print_structures(structures: List[Dict[str, Any]]) -> None:
    if not structures:
        return
    lines = ["\nSTRUCTURE SNAPSHOT"]
    for s in structures:
        lines.append(
            f"{s['signal_source']} | Price={s['current_price']:.2f} | "
            f"LongLower={s['long_lower']:.2f} | ShortUpper={s['short_upper']:.2f} | "
            f"Spacing={s['entry_spacing_dynamic']:.2f} | Reset={s['reset_trigger_dynamic']:.2f}"
        )
    logger.info("\n".join(lines))

# =============================================================================
# ANALYTICS / BACKTEST METRICS
# =============================================================================

def compute_equity_curve(trade_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not trade_logs:
        return []

    ordered = sorted(trade_logs, key=lambda x: pd.Timestamp(x["exit_time"]))
    equity = 0.0
    curve = []
    for t in ordered:
        equity += float(t["pnl"])
        curve.append({
            "exit_time": pd.Timestamp(t["exit_time"]),
            "equity": float(equity),
            "pnl": float(t["pnl"]),
        })
    return curve


def compute_max_drawdown(equity_values: Any) -> float:
    if len(equity_values) == 0:
        return 0.0
    values = np.array(equity_values, dtype=float)
    running_max = np.maximum.accumulate(values)
    drawdowns = values - running_max
    return float(drawdowns.min())


def compute_sharpe(returns: Any) -> float:
    returns = np.array(returns, dtype=float)
    if len(returns) == 0:
        return 0.0
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    val = returns.mean() / std
    if np.isnan(val) or np.isinf(val):
        return 0.0
    return float(val)


def compute_sortino(returns: Any) -> float:
    returns = np.array(returns, dtype=float)
    if len(returns) == 0:
        return 0.0
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return 0.0
    downside_std = negative_returns.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    val = returns.mean() / downside_std
    if np.isnan(val) or np.isinf(val):
        return 0.0
    return float(val)


def compute_cagr(trade_logs: List[Dict[str, Any]], ending_equity: float) -> float:
    if not trade_logs:
        return 0.0

    ordered = sorted(trade_logs, key=lambda x: pd.Timestamp(x["exit_time"]))
    start_time = pd.Timestamp(ordered[0]["entry_time"])
    end_time = pd.Timestamp(ordered[-1]["exit_time"])
    years = (end_time - start_time).total_seconds() / (86400 * 365.25)
    if years <= 0:
        return 0.0

    starting_equity = 1.0
    final_equity = starting_equity + float(ending_equity)
    if final_equity <= 0:
        return 0.0

    try:
        cagr = (final_equity / starting_equity) ** (1 / years) - 1
        if np.isnan(cagr) or np.isinf(cagr):
            return 0.0
        return float(cagr)
    except Exception:
        return 0.0


def print_trade_logs(trade_logs: List[Dict[str, Any]]) -> None:
    if not trade_logs:
        print("\nNo Closed Trades.")
        return

    ordered = sorted(trade_logs, key=lambda x: pd.Timestamp(x["exit_time"]))
    rows = []
    for idx, t in enumerate(ordered, 1):
        duration_days = (
            pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])
        ).total_seconds() / 86400
        rows.append({
            "#": idx,
            "TYPE": t["trade_type"],
            "ENTRY TIME": format_ts(t["entry_time"]),
            "EXIT TIME": format_ts(t["exit_time"]),
            "ENTRY PX": round(float(t["entry_price"]), 4),
            "EXIT PX": round(float(t["exit_price"]), 4),
            "SIZE": t.get("size", ""),
            "DAYS": round(duration_days, 4),
            "PNL": round(float(t["pnl"]), 4),
        })

    table = pd.DataFrame(rows)
    print("\n" + "=" * 140)
    print("TRADE LOGS")
    print("=" * 140)
    print(table.to_string(index=False, justify="center", col_space=12))
    print("=" * 140)
    print(
        f"TOTAL TRADES: {len(table)} | "
        f"TOTAL PNL: {round(table['PNL'].sum(), 4)} | "
        f"AVG PNL: {round(table['PNL'].mean(), 4)}"
    )
    print("=" * 140)


def plot_equity_curve(equity_curve: List[Dict[str, Any]]) -> None:
    if not equity_curve:
        return

    ordered = sorted(equity_curve, key=lambda x: pd.Timestamp(x["exit_time"]))
    x = [pd.Timestamp(p["exit_time"]) for p in ordered]
    y = [float(p["equity"]) for p in ordered]

    plt.figure(figsize=(14, 7))
    plt.plot(x, y)
    plt.title("AuctionBot BTC Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity / PnL Units")
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# =============================================================================
# BACKTEST ENGINE — QQQ AUCTIONBOT STRATEGY ON BTC DATA
# =============================================================================

def run_single_strategy(
    df: pd.DataFrame,
    long_anchor: int,
    long_entry_std: float,
    long_exit_std: float,
    short_anchor: int,
    short_entry_std: float,
    short_exit_std: float,
) -> Optional[Dict[str, Any]]:
    active_cycles: List[Trade] = []
    completed_cycles: List[Trade] = []
    recovery_cycles: List[Dict[str, Any]] = []

    last_long_entry: Optional[float] = None
    last_short_entry: Optional[float] = None
    local_long_armed = False
    local_short_armed = False

    lower_band_breaches = 0
    upper_band_breaches = 0
    long_cycles_triggered = 0
    recovery_cycles_triggered = 0
    reject_max_long = 0
    reject_max_short = 0
    reject_distance = 0
    reject_bias = 0
    reject_missing_anchor = 0

    df = prepare_strategy_dataframe(df, long_anchor, long_entry_std, short_anchor, short_entry_std)

    def evaluate_inventory() -> Tuple[int, int]:
        longs = sum(1 for t in active_cycles if not t.closed and t.side == "LONG")
        shorts = sum(1 for t in active_cycles if not t.closed and t.side == "SHORT")
        return longs, shorts

    def validate_direction(side: str) -> bool:
        longs, shorts = evaluate_inventory()
        if side == "LONG":
            return True
        return (longs - (shorts + 1)) >= DIRECTION_GATE

    def validate_spacing(last: Optional[float], price: float, spacing: float) -> bool:
        if last is None:
            return True
        if np.isnan(spacing):
            spacing = compute_adaptive_min_entry_distance(price)
        return abs(float(price) - float(last)) >= float(spacing)

    def resolve_anchor(ts: Any, current_price: float) -> Optional[Trade]:
        candidate = None
        ts = pd.Timestamp(ts)
        for t in active_cycles:
            if t.closed or t.side != "LONG" or t.inventory_lock:
                continue
            if t.entry_time >= ts:
                continue
            if current_price <= t.entry_price:
                continue
            if candidate is None or t.entry_time > candidate.entry_time:
                candidate = t
        return candidate

    def process_exits(price: float, ts: Any) -> None:
        ts = pd.Timestamp(ts)
        for t in active_cycles:
            if t.closed:
                continue
            if t.side == "LONG":
                if t.inventory_lock:
                    continue
                if price >= t.target:
                    t.exit_price = float(price)
                    t.exit_time = ts
                    t.pnl = (float(price) - t.entry_price) * t.size
                    t.closed = True
                    completed_cycles.append(t)
            else:
                if price <= t.target:
                    hedge = t.counter_leg
                    t.exit_price = float(price)
                    t.exit_time = ts
                    t.pnl = (t.entry_price - float(price)) * t.size
                    t.closed = True
                    completed_cycles.append(t)
                    if hedge is not None:
                        hedge.inventory_lock = False
                        hedge.recovery_owner = None

    def process_recovery(price: float, ts: Any) -> None:
        ts = pd.Timestamp(ts)
        for short in active_cycles:
            if short.closed or short.side != "SHORT":
                continue
            hedge = short.counter_leg
            if hedge is None:
                continue
            if hedge.recovery_owner != short:
                continue
            if hedge.closed or not hedge.inventory_lock:
                continue

            duration = (ts - short.entry_time).total_seconds() / 86400
            short_raw_pnl = short.entry_price - float(price)
            hedge_raw_pnl = float(price) - hedge.entry_price
            short_pnl = short_raw_pnl * short.size
            hedge_pnl = hedge_raw_pnl * hedge.size
            reset_trigger = float(short.reset_trigger if short.reset_trigger is not None else MANUAL_RESET_TRIGGER)

            if duration <= RESET_HORIZON:
                continue
            if short_raw_pnl >= reset_trigger:
                continue

            short.exit_price = float(price)
            short.exit_time = ts
            short.pnl = short_pnl
            short.closed = True

            hedge.exit_price = float(price)
            hedge.exit_time = ts
            hedge.pnl = hedge_pnl
            hedge.closed = True
            hedge.inventory_lock = False
            hedge.recovery_owner = None

            short.is_hedge_component = True
            hedge.is_hedge_component = True

            recovery_cycles.append({
                "entry_time": min(short.entry_time, hedge.entry_time),
                "exit_time": ts,
                "entry_price": (short.entry_price + hedge.entry_price) / 2,
                "exit_price": float(price),
                "long_size": float(hedge.size),
                "short_size": float(short.size),
                "size": f"L={hedge.size}/S={short.size}",
                "pnl": short_pnl + hedge_pnl,
            })
            return

    prices = df["close"].values.astype(float)
    timestamps = df["timestamp"].values
    prev = float(prices[0])

    for i in range(1, len(prices)):
        price = float(prices[i])
        ts = timestamps[i]

        process_exits(price, ts)
        process_recovery(price, ts)

        longs, shorts = evaluate_inventory()

        if not np.isnan(df["long_lower"].iloc[i]) and price < float(df["long_lower"].iloc[i]):
            lower_band_breaches += 1
            local_long_armed = True

        if local_long_armed and price > prev:
            rejected = False
            if longs >= MAX_LONG_POSITIONS:
                reject_max_long += 1
                rejected = True

            spacing = float(df["entry_spacing_dynamic"].iloc[i])
            if not validate_spacing(last_long_entry, price, spacing):
                reject_distance += 1
                rejected = True

            if not rejected:
                target = float(df["long_vwap"].iloc[i] + long_exit_std * df["long_std"].iloc[i])
                t = Trade("LONG", price, pd.Timestamp(ts), target, LONG_TRADE_SIZE)
                active_cycles.append(t)
                last_long_entry = price
                long_cycles_triggered += 1
                local_long_armed = False

        if price >= float(df["long_lower"].iloc[i]) if not np.isnan(df["long_lower"].iloc[i]) else False:
            local_long_armed = False

        if not np.isnan(df["short_upper"].iloc[i]) and price > float(df["short_upper"].iloc[i]):
            upper_band_breaches += 1
            local_short_armed = True

        if local_short_armed and price < prev:
            rejected = False
            if shorts >= MAX_SHORT_POSITIONS:
                reject_max_short += 1
                rejected = True

            spacing = float(df["entry_spacing_dynamic"].iloc[i])
            if not validate_spacing(last_short_entry, price, spacing):
                reject_distance += 1
                rejected = True

            if not validate_direction("SHORT"):
                reject_bias += 1
                rejected = True

            hedge = resolve_anchor(ts, price)
            if hedge is None:
                reject_missing_anchor += 1
                rejected = True

            if not rejected and hedge is not None:
                target = float(df["short_vwap"].iloc[i] - short_exit_std * df["short_std"].iloc[i])
                reset_trigger = float(df["reset_trigger_dynamic"].iloc[i])
                t = Trade("SHORT", price, pd.Timestamp(ts), target, SHORT_TRADE_SIZE, reset_trigger=reset_trigger)
                hedge.inventory_lock = True
                hedge.recovery_owner = t
                t.counter_leg = hedge
                active_cycles.append(t)
                last_short_entry = price
                recovery_cycles_triggered += 1
                local_short_armed = False

        if price <= float(df["short_upper"].iloc[i]) if not np.isnan(df["short_upper"].iloc[i]) else False:
            local_short_armed = False

        prev = price

    normal_pnls = [t.pnl for t in completed_cycles if not t.is_hedge_component]
    paired_pnls = [float(t["pnl"]) for t in recovery_cycles]
    pnls = np.array(normal_pnls + paired_pnls, dtype=float)

    if len(pnls) == 0:
        return None

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = safe_div(len(wins), len(pnls))
    avg_win = safe_mean(wins)
    avg_loss = abs(safe_mean(losses))
    profit_factor = np.inf if len(losses) == 0 else abs(wins.sum() / losses.sum())
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    trade_logs: List[Dict[str, Any]] = []
    for t in completed_cycles:
        if t.is_hedge_component:
            continue
        trade_logs.append({
            "trade_type": t.side,
            "entry_time": pd.Timestamp(t.entry_time),
            "exit_time": pd.Timestamp(t.exit_time),
            "entry_price": float(t.entry_price),
            "exit_price": float(t.exit_price),
            "size": float(t.size),
            "pnl": float(t.pnl),
        })

    for pt in recovery_cycles:
        trade_logs.append({
            "trade_type": "PAIRED",
            "entry_time": pd.Timestamp(pt["entry_time"]),
            "exit_time": pd.Timestamp(pt["exit_time"]),
            "entry_price": float(pt["entry_price"]),
            "exit_price": float(pt["exit_price"]),
            "size": pt.get("size", ""),
            "long_size": float(pt.get("long_size", 0.0)),
            "short_size": float(pt.get("short_size", 0.0)),
            "pnl": float(pt["pnl"]),
        })

    trade_logs = sorted(trade_logs, key=lambda x: pd.Timestamp(x["exit_time"]))
    equity_curve = compute_equity_curve(trade_logs)
    equity_values = [x["equity"] for x in equity_curve]

    durations = [
        (pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])).total_seconds() / 86400
        for t in trade_logs
    ]

    yearly_distribution: Dict[int, int] = {}
    for t in trade_logs:
        year = pd.Timestamp(t["exit_time"]).year
        yearly_distribution[year] = yearly_distribution.get(year, 0) + 1

    inside_bands = (df["close"] <= df["upper"]) & (df["close"] >= df["lower"])
    distance_from_vwap = (df["close"] - df["vwap"]).abs().dropna()
    std_stats = df["std"].dropna()
    lower_valid = df["lower"].first_valid_index()
    first_valid_timestamp = df.iloc[lower_valid]["timestamp"] if lower_valid is not None else None

    diagnostics = {
        "symbol": SYMBOL,
        "earliest_timestamp": df["timestamp"].min(),
        "latest_timestamp": df["timestamp"].max(),
        "total_rows": len(df),
        "warmup_period": max(long_anchor, short_anchor),
        "long_trade_size": float(LONG_TRADE_SIZE),
        "short_trade_size": float(SHORT_TRADE_SIZE),
        "first_valid_index": lower_valid,
        "first_valid_timestamp": first_valid_timestamp,
        "lower_band_breaches": lower_band_breaches,
        "upper_band_breaches": upper_band_breaches,
        "long_cycles_triggered": long_cycles_triggered,
        "recovery_cycles_triggered": recovery_cycles_triggered,
        "reject_max_long": reject_max_long,
        "reject_max_short": reject_max_short,
        "reject_distance": reject_distance,
        "reject_bias": reject_bias,
        "reject_missing_anchor": reject_missing_anchor,
        "avg_duration": safe_mean(durations),
        "median_duration": float(np.median(durations)) if len(durations) else 0.0,
        "max_duration": float(np.max(durations)) if len(durations) else 0.0,
        "yearly_distribution": yearly_distribution,
        "inside_band_pct": float(inside_bands.mean() * 100),
        "avg_distance_vwap": safe_mean(distance_from_vwap),
        "std_mean": safe_mean(std_stats),
        "std_median": float(np.median(std_stats)) if len(std_stats) else 0.0,
        "std_max": float(np.max(std_stats)) if len(std_stats) else 0.0,
        "adaptive_min_entry_distance_enabled": bool(USE_ADAPTIVE_MIN_ENTRY_DISTANCE),
        "adaptive_reset_trigger_enabled": bool(USE_ADAPTIVE_RESET_TRIGGER),
    }

    return {
        "trades": len(pnls),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "pnls": pnls,
        "trade_logs": trade_logs,
        "equity_curve": equity_curve,
        "sharpe": compute_sharpe(pnls),
        "sortino": compute_sortino(pnls),
        "max_drawdown": compute_max_drawdown(equity_values),
        "cagr": compute_cagr(trade_logs, equity_values[-1] if equity_values else 0.0),
        "diagnostics": diagnostics,
    }

# =============================================================================
# LIVE POSITION HELPERS
# =============================================================================

def init_statistics_template() -> Dict[str, Any]:
    return {
        "trade_counts_per_hour": {f"{h:02d}": 0 for h in range(24)},
        "largest_winner": 0.0,
        "largest_loser": 0.0,
        "total_trades": 0,
        "winning_trades": 0,
        "win_rate": 0.0,
        "sum_positive_pnl": 0.0,
        "sum_negative_pnl": 0.0,
        "profit_factor": 0.0,
        "sum_trade_durations": 0.0,
        "avg_hold_time_seconds": 0.0,
        "last_closed_trade_time": None,
    }


def bootstrap_files() -> None:
    if not os.path.exists(TRADE_LOG_FILE):
        write_json(TRADE_LOG_FILE, [])
    if not os.path.exists(OPENED_TRADES_FILE):
        write_json(OPENED_TRADES_FILE, [])
    if not os.path.exists(LIVE_STATUS_FILE):
        write_json(LIVE_STATUS_FILE, {"current_position": None, "cumulative_pnl": 0.0})
    if not os.path.exists(CURRENT_PRICE_FILE):
        write_json(CURRENT_PRICE_FILE, {"current_price": None})
    if not os.path.exists(STATISTICS_FILE):
        write_json(STATISTICS_FILE, init_statistics_template())
    if not os.path.exists(PNL_SUMMARY_FILE):
        write_json(PNL_SUMMARY_FILE, {
            "closed_pnl": 0.0,
            "open_pnl": 0.0,
            "net_pnl": 0.0,
            "current_price": None,
            "open_positions": 0,
            "long_positions": 0,
            "short_positions": 0,
            "net_exposure": 0.0,
            "gross_exposure": 0.0,
            "updated_at_est": est_now_str(),
            "update_frequency_seconds": int(PNL_SUMMARY_UPDATE_SECONDS),
            "engine": "BTC_AUCTIONBOT_ROLLING_VWAP_PAIRED",
        })


def initialize_open_positions() -> None:
    global open_positions
    data = read_json(OPENED_TRADES_FILE, default=[])
    if not isinstance(data, list):
        data = []
    open_positions = [p for p in data if p.get("status") == "OPEN"]
    logger.info(
        f"Restored {len(open_positions)} open position(s) - "
        f"Long: {count_open_positions_by_direction('LONG')}, "
        f"Short: {count_open_positions_by_direction('SHORT')}"
    )


def persist_open_positions() -> None:
    write_json(OPENED_TRADES_FILE, open_positions)


def get_open_positions(side: Optional[str] = None) -> List[Dict[str, Any]]:
    positions = [p for p in open_positions if p.get("status") == "OPEN"]
    if side is None:
        return positions
    return [p for p in positions if p.get("position") == side]


def count_open_positions_by_direction(direction: str) -> int:
    return len(get_open_positions(direction))


def get_position_by_id(trade_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if trade_id is None:
        return None
    for pos in open_positions:
        if pos.get("trade_id") == trade_id and pos.get("status") == "OPEN":
            return pos
    return None


def latest_open_entry_price(side: str) -> Optional[float]:
    positions = get_open_positions(side)
    if not positions:
        return None
    positions = sorted(positions, key=lambda p: pd.Timestamp(p.get("entry_time")))
    return float(positions[-1]["entry_price"])


def validate_spacing_last(side: str, price: float, spacing: float) -> bool:
    last = latest_open_entry_price(side)
    if last is None:
        return True
    return abs(float(price) - float(last)) >= float(spacing)


def validate_direction(side: str) -> bool:
    if side == "LONG":
        return True
    longs = count_open_positions_by_direction("LONG")
    shorts = count_open_positions_by_direction("SHORT")
    return (longs - (shorts + 1)) >= DIRECTION_GATE


def resolve_anchor(current_price: float) -> Optional[Dict[str, Any]]:
    candidate = None
    for pos in open_positions:
        if pos.get("status") != "OPEN":
            continue
        if pos.get("position") != "LONG":
            continue
        if pos.get("inventory_lock"):
            continue
        if float(current_price) <= float(pos.get("entry_price", 0.0)):
            continue
        if candidate is None:
            candidate = pos
            continue
        if pd.Timestamp(pos.get("entry_time")) > pd.Timestamp(candidate.get("entry_time")):
            candidate = pos
    return candidate


def compute_open_pnl(price: float) -> float:
    pnl = 0.0
    for pos in get_open_positions():
        entry = float(pos.get("entry_price", 0.0))
        size = float(pos.get("size", 1.0))
        if pos.get("position") == "LONG":
            pnl += (float(price) - entry) * size
        elif pos.get("position") == "SHORT":
            pnl += (entry - float(price)) * size
    return float(pnl)


def compute_closed_pnl() -> float:
    trade_log = read_json(TRADE_LOG_FILE, default=[])
    if not isinstance(trade_log, list):
        return 0.0
    return float(sum(float(t.get("pnl", 0.0)) for t in trade_log))

# =============================================================================
# BROKER / EMAIL DISPATCH
# =============================================================================

def format_trade_email(direction: str, action: str, symbol: str, qty: Any, price: float) -> str:
    now_est = datetime.now(tz_est).strftime("%Y-%m-%d %H:%M:%S EST")
    direction_text = direction.upper()
    action_text = action.upper()
    side_emoji = "🟢" if action.lower() in ["buy", "cover"] else "🔴"

    return f"""
═══════════════════════════════════════
            TRADE ALERT 🚨
═══════════════════════════════════════

{side_emoji} SIGNAL DETAILS

┌────────────────────┬────────────────────┐
│ Time (EST)         │ {now_est}
│ Direction          │ {direction_text}
│ Action             │ {action_text}
│ Signal Price       │ {price}
│ Symbol             │ {symbol}
│ Quantity           │ {qty}
└────────────────────┴────────────────────┘

═══════════════════════════════════════
System: BTC AuctionBot Rolling VWAP Paired Strategy
═══════════════════════════════════════
""".strip()


def send_email_alert(direction: str, action: str, symbol: str, qty: Any, price: float) -> None:
    if not ENABLE_EMAIL_ALERTS:
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["Subject"] = f"Trade Alert: {direction.upper()} {action.upper()} {symbol}"
        msg.attach(MIMEText(format_trade_email(direction, action, symbol, qty, price), "plain"))

        server = smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        for recipient in EMAIL_RECIPIENTS:
            msg["To"] = recipient
            server.sendmail(EMAIL_SENDER, recipient, msg.as_string())
            del msg["To"]
        server.quit()
        logger.info("Email alert sent successfully")
    except Exception as exc:
        logger.error(f"Email alert failed: {exc}")


def send_trade_signal(direction: str, action: str, qty: Any) -> Any:
    response = send_broker_order(
        env=ENV,
        direction=direction,
        action=action,
        qty=qty,
        send_orders_enabled=SEND_BROKER_ORDERS,
        logger=logger,
    )
    logger.info(
        f"Broker signal routed | backend={BROKER_BACKEND} | "
        f"direction={direction.upper()} action={action.upper()} qty={qty} | response={response}"
    )
    return response


def dispatch_trade_event(direction: str, action: str, qty: Any, price: float) -> None:
    symbol = BROKER_SYMBOL_LONG if direction == "long" else BROKER_SYMBOL_SHORT
    send_trade_signal(direction, action, qty)
    if ENABLE_EMAIL_ALERTS:
        send_email_alert(direction, action, symbol, qty, price)
    logger.info(
        f"TRADE EVENT | {direction.upper()} {action.upper()} {symbol} x{qty} | "
        f"broker={SEND_BROKER_ORDERS} email={ENABLE_EMAIL_ALERTS}"
    )

# =============================================================================
# LIVE TRADE LOGGING / STATS
# =============================================================================

def update_statistics_for_closed_trade(trade: Dict[str, Any]) -> None:
    stats = read_json(STATISTICS_FILE, default=init_statistics_template())
    if not isinstance(stats, dict):
        stats = init_statistics_template()

    pnl = float(trade.get("pnl", 0.0))
    stats["total_trades"] = int(stats.get("total_trades", 0)) + 1

    exit_time_text = trade.get("exit_time")
    if exit_time_text:
        try:
            exit_hour = datetime.strptime(exit_time_text, "%Y-%m-%d %H:%M:%S").strftime("%H")
            stats.setdefault("trade_counts_per_hour", {f"{h:02d}": 0 for h in range(24)})
            stats["trade_counts_per_hour"][exit_hour] = stats["trade_counts_per_hour"].get(exit_hour, 0) + 1
        except Exception:
            pass

    if pnl > 0:
        stats["winning_trades"] = int(stats.get("winning_trades", 0)) + 1
        stats["sum_positive_pnl"] = float(stats.get("sum_positive_pnl", 0.0)) + pnl
        stats["largest_winner"] = max(float(stats.get("largest_winner", 0.0)), pnl)
    else:
        stats["sum_negative_pnl"] = float(stats.get("sum_negative_pnl", 0.0)) + pnl
        stats["largest_loser"] = min(float(stats.get("largest_loser", 0.0)), pnl)

    total = int(stats.get("total_trades", 0))
    wins = int(stats.get("winning_trades", 0))
    stats["win_rate"] = (wins / total) * 100.0 if total else 0.0

    neg = float(stats.get("sum_negative_pnl", 0.0))
    pos = float(stats.get("sum_positive_pnl", 0.0))
    if neg != 0:
        stats["profit_factor"] = pos / abs(neg)
    elif pos > 0:
        stats["profit_factor"] = float("inf")
    else:
        stats["profit_factor"] = 0.0

    try:
        entry = datetime.strptime(trade["entry_time"], "%Y-%m-%d %H:%M:%S")
        exit_ = datetime.strptime(trade["exit_time"], "%Y-%m-%d %H:%M:%S")
        stats["sum_trade_durations"] = float(stats.get("sum_trade_durations", 0.0)) + (exit_ - entry).total_seconds()
        stats["avg_hold_time_seconds"] = stats["sum_trade_durations"] / total if total else 0.0
    except Exception:
        pass

    stats["last_closed_trade_time"] = trade.get("exit_time")
    write_json(STATISTICS_FILE, stats)


def append_trade_log(trade: Dict[str, Any]) -> None:
    trade_log = read_json(TRADE_LOG_FILE, default=[])
    if not isinstance(trade_log, list):
        trade_log = []
    trade_log.append(trade)
    write_json(TRADE_LOG_FILE, trade_log)
    update_statistics_for_closed_trade(trade)

# =============================================================================
# LIVE ENTRY / EXIT ENGINE — SAME AUCTIONBOT LOGIC
# =============================================================================

def open_long(price: float, structure: Dict[str, Any]) -> bool:
    if count_open_positions_by_direction("LONG") >= MAX_LONG_POSITIONS:
        return False

    spacing = float(structure["entry_spacing_dynamic"])
    if not validate_spacing_last("LONG", price, spacing):
        return False

    target = float(structure["long_vwap"] + float(structure["long_exit_std"]) * float(structure["long_std"]))
    trade_id = f"LONG_{int(time.time() * 1000)}"
    trade = {
        "trade_id": trade_id,
        "position": "LONG",
        "entry_price": float(price),
        "entry_time": est_now_str(),
        "status": "OPEN",
        "size": float(LONG_TRADE_SIZE),
        "broker_qty": BROKER_ORDER_SIZE_LONG,
        "target": target,
        "target_mode": "FIXED_ROLLING_VWAP",
        "inventory_lock": False,
        "recovery_owner": None,
        "counter_leg": None,
        "is_hedge_component": False,
        "signal_engine": "ROLLING_VWAP_AUCTIONBOT",
        "signal_source": structure.get("signal_source"),
        "long_anchor": int(structure.get("long_anchor")),
        "short_anchor": int(structure.get("short_anchor")),
        "entry_spacing": spacing,
        "reset_trigger": float(structure.get("reset_trigger_dynamic")),
    }

    open_positions.append(trade)
    persist_open_positions()
    dispatch_trade_event("long", "buy", BROKER_ORDER_SIZE_LONG, price)
    logger.info(
        f"LONG OPENED | Engine=AUCTIONBOT_ROLLING_VWAP | Source={trade['signal_source']} | "
        f"Price={price:.2f} | Target={target:.2f} | StrategySize={LONG_TRADE_SIZE} | Spacing={spacing:.2f}"
    )
    return True


def open_short(price: float, structure: Dict[str, Any], hedge: Dict[str, Any]) -> bool:
    if count_open_positions_by_direction("SHORT") >= MAX_SHORT_POSITIONS:
        return False

    spacing = float(structure["entry_spacing_dynamic"])
    if not validate_spacing_last("SHORT", price, spacing):
        return False

    if not validate_direction("SHORT"):
        return False

    target = float(structure["short_vwap"] - float(structure["short_exit_std"]) * float(structure["short_std"]))
    reset_trigger = float(structure["reset_trigger_dynamic"])
    trade_id = f"SHORT_{int(time.time() * 1000)}"

    trade = {
        "trade_id": trade_id,
        "position": "SHORT",
        "entry_price": float(price),
        "entry_time": est_now_str(),
        "status": "OPEN",
        "size": float(SHORT_TRADE_SIZE),
        "broker_qty": BROKER_ORDER_SIZE_SHORT,
        "target": target,
        "target_mode": "FIXED_ROLLING_VWAP",
        "inventory_lock": False,
        "recovery_owner": None,
        "counter_leg": hedge.get("trade_id"),
        "is_hedge_component": False,
        "signal_engine": "ROLLING_VWAP_AUCTIONBOT_PAIRED_SHORT",
        "signal_source": structure.get("signal_source"),
        "long_anchor": int(structure.get("long_anchor")),
        "short_anchor": int(structure.get("short_anchor")),
        "entry_spacing": spacing,
        "reset_trigger": reset_trigger,
        "paired_long_entry_price": float(hedge.get("entry_price")),
        "paired_long_trade_id": hedge.get("trade_id"),
    }

    hedge["inventory_lock"] = True
    hedge["recovery_owner"] = trade_id
    hedge["counter_leg"] = trade_id

    open_positions.append(trade)
    persist_open_positions()
    dispatch_trade_event("short", "sell", BROKER_ORDER_SIZE_SHORT, price)
    logger.info(
        f"SHORT OPENED | Engine=AUCTIONBOT_PAIRED_SHORT | Source={trade['signal_source']} | "
        f"Price={price:.2f} | Target={target:.2f} | StrategySize={SHORT_TRADE_SIZE} | "
        f"HedgeLong={hedge.get('trade_id')} | Reset={reset_trigger:.2f}"
    )
    return True


def close_position_standard(pos: Dict[str, Any], price: float, exit_reason: str) -> Dict[str, Any]:
    side = pos.get("position")
    entry = float(pos.get("entry_price", 0.0))
    size = float(pos.get("size", 1.0))

    if side == "LONG":
        pnl = (float(price) - entry) * size
        direction = "long"
        action = "sell"
        qty = pos.get("broker_qty", BROKER_ORDER_SIZE_LONG)
        trade_type = "LONG"
    else:
        pnl = (entry - float(price)) * size
        direction = "short"
        action = "cover"
        qty = pos.get("broker_qty", BROKER_ORDER_SIZE_SHORT)
        trade_type = "SHORT"

    pos["status"] = "CLOSED"
    pos["exit_price"] = float(price)
    pos["exit_time"] = est_now_str()
    pos["pnl"] = float(pnl)
    pos["exit_reason"] = exit_reason

    if side == "SHORT":
        hedge = get_position_by_id(pos.get("counter_leg"))
        if hedge is not None:
            hedge["inventory_lock"] = False
            hedge["recovery_owner"] = None
            hedge["counter_leg"] = None

    log_entry = {
        "trade_id": pos.get("trade_id"),
        "trade_type": trade_type,
        "entry_time": pos.get("entry_time"),
        "exit_time": pos.get("exit_time"),
        "entry_price": entry,
        "exit_price": float(price),
        "size": size,
        "broker_qty": qty,
        "pnl": float(pnl),
        "exit_reason": exit_reason,
        "signal_engine": pos.get("signal_engine"),
        "signal_source": pos.get("signal_source"),
    }

    append_trade_log(log_entry)
    dispatch_trade_event(direction, action, qty, price)
    persist_open_positions()
    logger.info(
        f"{trade_type} CLOSED | Reason={exit_reason} | Entry={entry:.2f} | "
        f"Exit={price:.2f} | Size={size} | PnL={pnl:.2f}"
    )
    return log_entry


def process_standard_exits(price: float) -> None:
    for pos in list(get_open_positions()):
        side = pos.get("position")
        target = float(pos.get("target", 0.0))

        if side == "LONG":
            if pos.get("inventory_lock"):
                continue
            if float(price) >= target:
                close_position_standard(pos, price, "TARGET")

        elif side == "SHORT":
            if float(price) <= target:
                close_position_standard(pos, price, "TARGET")

    open_positions[:] = [p for p in open_positions if p.get("status") == "OPEN"]
    persist_open_positions()


def close_recovery_pair(short: Dict[str, Any], hedge: Dict[str, Any], price: float) -> Dict[str, Any]:
    now = est_now_str()

    short_entry = float(short.get("entry_price", 0.0))
    hedge_entry = float(hedge.get("entry_price", 0.0))
    short_size = float(short.get("size", SHORT_TRADE_SIZE))
    hedge_size = float(hedge.get("size", LONG_TRADE_SIZE))

    short_raw_pnl = short_entry - float(price)
    hedge_raw_pnl = float(price) - hedge_entry
    short_pnl = short_raw_pnl * short_size
    hedge_pnl = hedge_raw_pnl * hedge_size
    total_pnl = short_pnl + hedge_pnl

    short["status"] = "CLOSED"
    short["exit_price"] = float(price)
    short["exit_time"] = now
    short["pnl"] = float(short_pnl)
    short["is_hedge_component"] = True
    short["exit_reason"] = "PAIR_RESET_RECOVERY"

    hedge["status"] = "CLOSED"
    hedge["exit_price"] = float(price)
    hedge["exit_time"] = now
    hedge["pnl"] = float(hedge_pnl)
    hedge["inventory_lock"] = False
    hedge["recovery_owner"] = None
    hedge["counter_leg"] = None
    hedge["is_hedge_component"] = True
    hedge["exit_reason"] = "PAIR_RESET_RECOVERY"

    log_entry = {
        "trade_type": "PAIRED",
        "short_trade_id": short.get("trade_id"),
        "long_trade_id": hedge.get("trade_id"),
        "entry_time": min(str(short.get("entry_time")), str(hedge.get("entry_time"))),
        "exit_time": now,
        "entry_price": (short_entry + hedge_entry) / 2.0,
        "exit_price": float(price),
        "size": f"L={hedge_size}/S={short_size}",
        "long_size": hedge_size,
        "short_size": short_size,
        "long_pnl": float(hedge_pnl),
        "short_pnl": float(short_pnl),
        "pnl": float(total_pnl),
        "exit_reason": "PAIR_RESET_RECOVERY",
        "signal_engine": "ROLLING_VWAP_AUCTIONBOT_PAIR_RESET",
        "signal_source": short.get("signal_source"),
    }

    append_trade_log(log_entry)
    dispatch_trade_event("short", "cover", short.get("broker_qty", BROKER_ORDER_SIZE_SHORT), price)
    dispatch_trade_event("long", "sell", hedge.get("broker_qty", BROKER_ORDER_SIZE_LONG), price)

    open_positions[:] = [p for p in open_positions if p.get("status") == "OPEN"]
    persist_open_positions()
    logger.info(
        f"PAIR RESET CLOSED | Short={short.get('trade_id')} Long={hedge.get('trade_id')} | "
        f"Price={price:.2f} | PnL={total_pnl:.2f} | ShortRaw={short_raw_pnl:.2f}"
    )
    return log_entry


def process_recovery_exits(price: float) -> None:
    for short in list(get_open_positions("SHORT")):
        hedge = get_position_by_id(short.get("counter_leg"))
        if hedge is None:
            continue
        if hedge.get("recovery_owner") != short.get("trade_id"):
            continue
        if not hedge.get("inventory_lock"):
            continue

        try:
            short_entry_time = pd.Timestamp(short.get("entry_time"))
        except Exception:
            continue

        duration_days = (pd.Timestamp(est_now_str()) - short_entry_time).total_seconds() / 86400
        short_raw_pnl = float(short.get("entry_price", 0.0)) - float(price)
        reset_trigger = float(short.get("reset_trigger", MANUAL_RESET_TRIGGER))

        if duration_days <= RESET_HORIZON:
            continue
        if short_raw_pnl >= reset_trigger:
            continue

        close_recovery_pair(short, hedge, price)
        return


def process_entries(price: float, structure: Dict[str, Any]) -> None:
    global previous_price, long_armed, short_armed, long_arm_extreme, short_arm_extreme

    if previous_price is None:
        previous_price = float(price)
        return

    price = float(price)
    prev = float(previous_price)

    if price < float(structure["long_lower"]):
        long_armed = True
        long_arm_extreme = price if long_arm_extreme is None else min(long_arm_extreme, price)
        log_message_throttled(
            f"LONG ARMED | Price={price:.2f} < LongLower={structure['long_lower']:.2f} | Extreme={long_arm_extreme:.2f}",
            "long_armed",
        )

    if long_armed and price > prev:
        opened = open_long(price, structure)
        if opened:
            long_armed = False
            long_arm_extreme = None

    if price >= float(structure["long_lower"]):
        long_armed = False
        long_arm_extreme = None

    if price > float(structure["short_upper"]):
        short_armed = True
        short_arm_extreme = price if short_arm_extreme is None else max(short_arm_extreme, price)
        log_message_throttled(
            f"SHORT ARMED | Price={price:.2f} > ShortUpper={structure['short_upper']:.2f} | Extreme={short_arm_extreme:.2f}",
            "short_armed",
        )

    if short_armed and price < prev:
        hedge = resolve_anchor(price)
        if hedge is None:
            log_message_throttled(
                "SHORT BLOCKED | No profitable unlocked long anchor available for paired short.",
                "short_no_anchor",
            )
        else:
            opened = open_short(price, structure, hedge)
            if opened:
                short_armed = False
                short_arm_extreme = None

    if price <= float(structure["short_upper"]):
        short_armed = False
        short_arm_extreme = None

# =============================================================================
# LIVE STATUS / PNL SUMMARY
# =============================================================================

def update_live_status(price: float, structure: Optional[Dict[str, Any]] = None) -> None:
    closed_pnl = compute_closed_pnl()
    open_pnl = compute_open_pnl(price)
    longs = count_open_positions_by_direction("LONG")
    shorts = count_open_positions_by_direction("SHORT")

    write_json(LIVE_STATUS_FILE, {
        "engine": "BTC_AUCTIONBOT_ROLLING_VWAP_PAIRED",
        "symbol": SYMBOL,
        "current_price": float(price),
        "open_positions": len(get_open_positions()),
        "long_positions": longs,
        "short_positions": shorts,
        "net_exposure_count": longs - shorts,
        "gross_exposure_count": longs + shorts,
        "strategy_long_trade_size": float(LONG_TRADE_SIZE),
        "strategy_short_trade_size": float(SHORT_TRADE_SIZE),
        "broker_order_size_long": BROKER_ORDER_SIZE_LONG,
        "broker_order_size_short": BROKER_ORDER_SIZE_SHORT,
        "bias_direction": BIAS_DIRECTION,
        "direction_gate": DIRECTION_GATE,
        "long_armed": bool(long_armed),
        "short_armed": bool(short_armed),
        "long_arm_extreme": long_arm_extreme,
        "short_arm_extreme": short_arm_extreme,
        "closed_pnl": float(closed_pnl),
        "open_pnl": float(open_pnl),
        "net_pnl": float(closed_pnl + open_pnl),
        "latest_structure": structure or latest_market_structure,
        "timestamp_est": est_now_str(),
    })


def update_pnl_summary(price: float, force: bool = False) -> None:
    global last_pnl_summary_update

    now = utc_now()
    if not force and last_pnl_summary_update is not None:
        elapsed = (now - last_pnl_summary_update).total_seconds()
        if elapsed < float(PNL_SUMMARY_UPDATE_SECONDS):
            return

    closed_pnl = compute_closed_pnl()
    open_pnl = compute_open_pnl(price)
    longs = count_open_positions_by_direction("LONG")
    shorts = count_open_positions_by_direction("SHORT")

    summary = {
        "engine": "BTC_AUCTIONBOT_ROLLING_VWAP_PAIRED",
        "symbol": SYMBOL,
        "closed_pnl": float(closed_pnl),
        "open_pnl": float(open_pnl),
        "net_pnl": float(closed_pnl + open_pnl),
        "current_price": float(price),
        "open_positions": len(get_open_positions()),
        "long_positions": longs,
        "short_positions": shorts,
        "net_exposure": longs - shorts,
        "gross_exposure": longs + shorts,
        "strategy_long_trade_size": float(LONG_TRADE_SIZE),
        "strategy_short_trade_size": float(SHORT_TRADE_SIZE),
        "updated_at_est": est_now_str(),
        "update_frequency_seconds": int(PNL_SUMMARY_UPDATE_SECONDS),
    }
    write_json(PNL_SUMMARY_FILE, summary)
    last_pnl_summary_update = now

# =============================================================================
# GRACEFUL SHUTDOWN
# =============================================================================

def graceful_shutdown(signum, frame) -> None:
    logger.info("Shutdown signal received. Exiting cleanly.")
    try:
        shutdown_broker(logger=logger)
    except Exception as exc:
        logger.error(f"Broker shutdown failed: {exc}")
    sys.exit(0)


signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)

# =============================================================================
# RUNNERS
# =============================================================================

def run_backtest() -> None:
    portfolio_pnls: List[float] = []
    print(f"\n================ {SYMBOL} BACKTEST ================")
    df = fetch_data(SYMBOL, START_DATE)
    if df is None or df.empty:
        print("No BTC data returned.")
        return

    for long_anchor in LONG_ANCHORS:
        for long_entry in LONG_ENTRY_STDS:
            for long_exit in LONG_EXIT_STDS:
                for short_anchor in SHORT_ANCHORS:
                    for short_entry in SHORT_ENTRY_STDS:
                        for short_exit in SHORT_EXIT_STDS:
                            print(
                                f"\nRunning | LONG_ANCHOR={long_anchor} | SHORT_ANCHOR={short_anchor} | "
                                f"LONG_ENTRY_STD={long_entry} | SHORT_ENTRY_STD={short_entry} | "
                                f"LONG_SIZE={LONG_TRADE_SIZE} | SHORT_SIZE={SHORT_TRADE_SIZE}"
                            )
                            result = run_single_strategy(
                                df.copy(),
                                long_anchor,
                                long_entry,
                                long_exit,
                                short_anchor,
                                short_entry,
                                short_exit,
                            )
                            if result is None:
                                print("No closed trades for this configuration.")
                                continue

                            print("Trades:", result["trades"])
                            print("Win Rate:", round(result["win_rate"] * 100, 2), "%")
                            print("PF:", result["profit_factor"])
                            print("Expectancy:", result["expectancy"])
                            print("Sharpe Ratio:", round(result["sharpe"], 6))
                            print("Sortino Ratio:", round(result["sortino"], 6))
                            print("Max Drawdown:", round(result["max_drawdown"], 6))
                            print("Diagnostics:", result["diagnostics"])
                            print_trade_logs(result["trade_logs"])
                            plot_equity_curve(result["equity_curve"])
                            portfolio_pnls.extend([float(x) for x in result["pnls"]])

    if portfolio_pnls:
        pnls = np.array(portfolio_pnls, dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        win_rate = safe_div(len(wins), len(pnls))
        pf = np.inf if len(losses) == 0 else abs(wins.sum() / losses.sum())
        print("\n================ PORTFOLIO ================")
        print("Trades:", len(pnls))
        print("Win Rate:", round(win_rate * 100, 2), "%")
        print("PF:", pf)
        print("Sharpe Ratio:", round(compute_sharpe(pnls), 6))
        print("Sortino Ratio:", round(compute_sortino(pnls), 6))
        print("Max Drawdown:", round(compute_max_drawdown(np.cumsum(pnls)), 6))


def run_trading_bot() -> None:
    global previous_price, latest_market_structure

    bootstrap_files()
    initialize_open_positions()
    initialize_broker(logger=logger, send_orders_enabled=SEND_BROKER_ORDERS)

    logger.info(
        f"AuctionBot BTC live runner started | symbol={SYMBOL} | broker_backend={BROKER_BACKEND} | "
        f"send_broker_orders={SEND_BROKER_ORDERS} | strategy_sizes=L{LONG_TRADE_SIZE}/S{SHORT_TRADE_SIZE}"
    )

    last_heartbeat = utc_now()
    last_structure_refresh: Optional[datetime] = None

    while True:
        try:
            now_utc = utc_now()
            if (now_utc - last_heartbeat).total_seconds() >= HEARTBEAT_FREQUENCY_MINUTES * 60:
                log_heartbeat()
                last_heartbeat = now_utc

            raw_price = fetch_latest_price()
            if raw_price is None:
                time.sleep(LIVE_LOOP_SLEEP_SECONDS)
                continue

            price = filter_live_price(raw_price)
            if price is None:
                time.sleep(LIVE_LOOP_SLEEP_SECONDS)
                continue

            write_json(CURRENT_PRICE_FILE, {"current_price": float(price), "timestamp_est": est_now_str()})
            log_message_throttled(f"Live price: {price:.2f}", "price")

            refresh_required = False
            if last_structure_refresh is None:
                refresh_required = True
            elif (now_utc - last_structure_refresh).total_seconds() >= VWAP_RECOMPUTE_INTERVAL_SECONDS:
                refresh_required = True

            if refresh_required:
                structures = compute_all_market_structures()
                if structures:
                    latest_market_structure = structures[0]
                    last_structure_refresh = now_utc
                    debug_print_structures(structures)
                    logger.info(
                        f"ACTIVE STRUCTURE={latest_market_structure['signal_source']} | "
                        f"LongLower={latest_market_structure['long_lower']:.2f} | "
                        f"ShortUpper={latest_market_structure['short_upper']:.2f}"
                    )
                else:
                    log_message_throttled("No valid market structure yet.", "no_structure")

            if not latest_market_structure:
                previous_price = float(price)
                time.sleep(LIVE_LOOP_SLEEP_SECONDS)
                continue

            if not is_restricted_hour_est():
                process_standard_exits(float(price))
                process_recovery_exits(float(price))
                process_entries(float(price), latest_market_structure)
            else:
                log_message_throttled("Restricted EST hour — trading paused", "restricted_hour")

            update_live_status(float(price), latest_market_structure)
            update_pnl_summary(float(price))
            previous_price = float(price)
            time.sleep(LIVE_LOOP_SLEEP_SECONDS)

        except Exception as exc:
            logger.exception(f"Unexpected runtime error: {exc}")
            time.sleep(LIVE_LOOP_SLEEP_SECONDS)


if __name__ == "__main__":
    if str(MODE).upper() == "BACKTEST":
        run_backtest()
    elif str(MODE).upper() == "LIVE":
        run_trading_bot()
    else:
        raise ValueError("MODE must be either 'LIVE' or 'BACKTEST'.")
