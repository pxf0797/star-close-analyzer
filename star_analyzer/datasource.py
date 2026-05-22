"""
数据源抽象层 — 双通道设计：合成数据 + 真实行情 API。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pandas as pd
import requests
import time


@dataclass
class DataMeta:
    name: str
    length: int
    start_time: str = ""
    end_time: str = ""
    source: str = ""


class DataSource(ABC):
    @abstractmethod
    def fetch(self, n: int = 1000) -> np.ndarray:
        """返回 close 价格数组"""
        ...

    @abstractmethod
    def fetch_df(self, n: int = 1000) -> pd.DataFrame:
        """返回完整 DataFrame (timestamp, open, high, low, close, volume)"""
        ...

    @property
    @abstractmethod
    def meta(self) -> DataMeta:
        ...


# ═══════════════════════════════════════════════
# 合成数据源
# ═══════════════════════════════════════════════

class SinSource(DataSource):
    """多周期正弦叠加 + 噪声 — 快速演示（当前默认方案）"""

    def __init__(self, n: int = 500, seed: int = 42):
        self.n = n
        self.seed = seed

    def fetch(self, n: int | None = None) -> np.ndarray:
        n = n or self.n
        rng = np.random.RandomState(self.seed)
        t = np.arange(n, dtype=float)
        trend1 = 100 + 10 * np.sin(2 * np.pi * t / 200)
        trend2 = 5 * np.sin(2 * np.pi * t / 60)
        trend3 = 2 * np.sin(2 * np.pi * t / 25)
        noise = rng.randn(n) * 1.5
        return np.maximum(trend1 + trend2 + trend3 + noise, 1.0)

    def fetch_df(self, n: int | None = None) -> pd.DataFrame:
        n = n or self.n
        prices = self.fetch(n)
        rng = np.random.RandomState(self.seed + 1)  # +1 避免与 fetch 共用相同种子序列
        df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=n, freq="1h"),
                           "close": prices})
        df["open"] = df["close"] * (1 + rng.randn(n) * 0.001)
        df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.abs(rng.randn(n) * 0.002))
        df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.abs(rng.randn(n) * 0.002))
        df["volume"] = rng.rand(n) * 100
        return df

    @property
    def meta(self) -> DataMeta:
        return DataMeta(name=f"正弦叠加 (n={self.n})", length=self.n, source="synthetic")


class PolySource(DataSource):
    """分段三次样条 + AR(1) 噪声 — 轨迹拟合专项测试。
    control_points: dict[int, tuple[float, str]] — {index: (price, "peak"|"valley")}
    若未提供则生成默认峰谷序列。
    """

    def __init__(self, control_points: dict | None = None, n: int = 500,
                 noise_rho: float = 0.5, noise_sigma: float = 1.0, seed: int = 42):
        self._control_points = control_points
        self.n = n
        self.noise_rho = noise_rho
        self.noise_sigma = noise_sigma
        self.seed = seed

    def fetch(self, n: int | None = None) -> np.ndarray:
        n = n or self.n
        rng = np.random.RandomState(self.seed)

        cp = self._control_points or self._default_control_points(n)
        xs = sorted(cp.keys())
        ys = [cp[x][0] for x in xs]

        from scipy.interpolate import CubicSpline
        cs = CubicSpline(xs, ys, bc_type="natural")
        t = np.arange(n, dtype=float)
        base = cs(t)

        # AR(1) 噪声
        noise = np.zeros(n)
        for i in range(1, n):
            noise[i] = self.noise_rho * noise[i - 1] + self.noise_sigma * rng.randn()

        return np.maximum(base + noise, 1.0)

    def _default_control_points(self, n: int) -> dict:
        cp = {}
        for i in range(0, n, 40):
            is_peak = (i // 40) % 2 == 0
            cp[i] = (100.0 + (10 if is_peak else -10) * ((i // 40 + 1) * 0.8), "peak" if is_peak else "valley")
        return cp

    def fetch_df(self, n: int | None = None) -> pd.DataFrame:
        n = n or self.n
        prices = self.fetch(n)
        rng = np.random.RandomState(self.seed + 1)
        df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=n, freq="1h"),
                           "close": prices})
        df["open"] = df["close"] * (1 + rng.randn(n) * 0.001)
        df["high"] = df[["open", "close"]].max(axis=1) * 1.002
        df["low"] = df[["open", "close"]].min(axis=1) * 0.998
        df["volume"] = rng.rand(n) * 100
        return df

    @property
    def meta(self) -> DataMeta:
        return DataMeta(name=f"分段样条+AR(1) (n={self.n})", length=self.n, source="synthetic")


class GbmSource(DataSource):
    """几何布朗运动 — 统计回测"""

    def __init__(self, n: int = 500, s0: float = 100.0, mu: float = 0.0,
                 sigma: float = 0.02, seed: int = 42):
        self.n = n
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.seed = seed

    def fetch(self, n: int | None = None) -> np.ndarray:
        n = n or self.n
        rng = np.random.RandomState(self.seed)
        dt = 1.0
        returns = self.mu * dt + self.sigma * rng.randn(n) * np.sqrt(dt)
        prices = self.s0 * np.exp(np.cumsum(returns))
        return np.maximum(prices, 1.0)

    def fetch_df(self, n: int | None = None) -> pd.DataFrame:
        n = n or self.n
        prices = self.fetch(n)
        rng = np.random.RandomState(self.seed + 1)
        df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=n, freq="1h"),
                           "close": prices})
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["high"] = df[["open", "close"]].max(axis=1) * 1.002
        df["low"] = df[["open", "close"]].min(axis=1) * 0.998
        df["volume"] = rng.rand(n) * 100
        return df

    @property
    def meta(self) -> DataMeta:
        return DataMeta(name=f"GBM (μ={self.mu}, σ={self.sigma})", length=self.n, source="synthetic")


# ═══════════════════════════════════════════════
# 真实行情 API
# ═══════════════════════════════════════════════

class KrakenSource(DataSource):
    """Kraken 公开 REST API — 首选免费数据源"""

    BASE = "https://api.kraken.com/0/public/OHLC"
    INTERVALS = {1: 1, 5: 5, 15: 15, 30: 30, 60: 60, 240: 240, 1440: 1440, 10080: 10080}

    def __init__(self, pair: str = "XBTUSD", interval: int = 60):
        self.pair = pair
        self.interval = interval
        self._cached_df: pd.DataFrame | None = None

    def fetch(self, n: int | None = None) -> np.ndarray:
        df = self.fetch_df()
        return df["close"].values.astype(float)

    def fetch_df(self, n: int | None = None) -> pd.DataFrame:
        if self._cached_df is not None:
            return self._cached_df

        params = {"pair": self.pair, "interval": self.interval}
        resp = requests.get(self.BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("error"):
            raise RuntimeError(f"Kraken API error: {data['error']}")

        ohlc = data["result"][list(data["result"].keys())[0]]
        rows = []
        for k in ohlc:
            rows.append({
                "timestamp": pd.to_datetime(int(k[0]), unit="s"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[6]),
            })

        df = pd.DataFrame(rows)
        self._cached_df = df
        return df

    @property
    def meta(self) -> DataMeta:
        df = self.fetch_df()
        return DataMeta(
            name=f"Kraken {self.pair} {self.interval}m",
            length=len(df),
            start_time=str(df["timestamp"].iloc[0]),
            end_time=str(df["timestamp"].iloc[-1]),
            source="kraken",
        )


class CoinGeckoSource(DataSource):
    """CoinGecko 公开 API — 已验证可用"""

    BASE = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

    def __init__(self, days: int = 30):
        self.days = days
        self._cached_df: pd.DataFrame | None = None

    def fetch(self, n: int | None = None) -> np.ndarray:
        df = self.fetch_df()
        return df["close"].values.astype(float)

    def fetch_df(self, n: int | None = None) -> pd.DataFrame:
        if self._cached_df is not None:
            return self._cached_df

        resp = requests.get(self.BASE, params={"vs_currency": "usd", "days": self.days}, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        prices = data["prices"]
        df = pd.DataFrame(prices, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp").resample("1h").agg({"close": "last"}).dropna().reset_index()

        df["open"] = df["close"]
        df["high"] = df["close"] * 1.002
        df["low"] = df["close"] * 0.998
        df["volume"] = 0

        self._cached_df = df
        return df

    @property
    def meta(self) -> DataMeta:
        df = self.fetch_df()
        return DataMeta(
            name=f"CoinGecko BTC/USD {self.days}d",
            length=len(df),
            start_time=str(df["timestamp"].iloc[0]),
            end_time=str(df["timestamp"].iloc[-1]),
            source="coingecko",
        )


class OkxSource(DataSource):
    """OKX 公开 REST API"""

    BASE = "https://www.okx.com/api/v5/market/candles"

    def __init__(self, inst_id: str = "BTC-USDT", bar: str = "1H", limit: int = 300):
        self.inst_id = inst_id
        self.bar = bar
        self.limit = limit
        self._cached_df: pd.DataFrame | None = None

    def fetch(self, n: int | None = None) -> np.ndarray:
        df = self.fetch_df()
        return df["close"].values.astype(float)

    def fetch_df(self, n: int | None = None) -> pd.DataFrame:
        if self._cached_df is not None:
            return self._cached_df

        params = {"instId": self.inst_id, "bar": self.bar, "limit": str(self.limit)}
        resp = requests.get(self.BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != "0":
            raise RuntimeError(f"OKX API error: {data.get('msg')}")

        rows = []
        for k in reversed(data["data"]):
            rows.append({
                "timestamp": pd.to_datetime(int(k[0]), unit="ms"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })

        df = pd.DataFrame(rows)
        self._cached_df = df
        return df

    @property
    def meta(self) -> DataMeta:
        df = self.fetch_df()
        return DataMeta(
            name=f"OKX {self.inst_id} {self.bar}",
            length=len(df),
            start_time=str(df["timestamp"].iloc[0]),
            end_time=str(df["timestamp"].iloc[-1]),
            source="okx",
        )


class CsvSource(DataSource):
    """CSV 文件数据源"""

    def __init__(self, filepath: str, col: str = "close"):
        self.filepath = filepath
        self.col = col
        self._cached_df: pd.DataFrame | None = None
        self._len: int | None = None

    def fetch(self, n: int | None = None) -> np.ndarray:
        df = self.fetch_df()
        return df[self.col].values.astype(float)

    def fetch_df(self, n: int | None = None) -> pd.DataFrame:
        if self._cached_df is not None:
            return self._cached_df
        df = pd.read_csv(self.filepath)
        self._cached_df = df
        return df

    @property
    def meta(self) -> DataMeta:
        if self._len is None:
            self._len = len(self.fetch_df())
        return DataMeta(name=f"CSV: {self.filepath}", length=self._len, source="csv")


# ═══════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════

SOURCE_REGISTRY = {
    "sin": lambda **kw: SinSource(**kw),
    "poly": lambda **kw: PolySource(**kw),
    "gbm": lambda **kw: GbmSource(**kw),
    "kraken": lambda **kw: KrakenSource(**kw),
    "coingecko": lambda **kw: CoinGeckoSource(**kw),
    "okx": lambda **kw: OkxSource(**kw),
}


def create_source(name: str, **kwargs) -> DataSource:
    if name in SOURCE_REGISTRY:
        return SOURCE_REGISTRY[name](**kwargs)
    raise ValueError(f"Unknown source: {name}. Available: {list(SOURCE_REGISTRY.keys())}")
