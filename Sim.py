# src/simulate_mc.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Callable, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class PortfolioConfig:
    weights: Dict[str, float]
    years: int
    n_sims: int = 10_000
    initial_balance: float = 0.0
    sample_mode: str = "iid" 
    block_size: int = 3
    seed: Optional[int] = 42


def load_market_inputs(xlsx_path: str | Path) -> dict:
    wb = pd.ExcelFile(xlsx_path)
    out = {
        "returns_annual": pd.read_excel(wb, "Returns_Annual", index_col=0),
    }
    try:
        cpi_annual = pd.read_excel(wb, "CPI_Annual", index_col=0)
        out["inflation_annual"] = cpi_annual["inflation_yoy"].dropna()
    except ValueError:
        out["inflation_annual"] = None
    return out


def _normalize_weights(w: Dict[str, float]) -> pd.Series:
    s = pd.Series(w, dtype=float)
    total = s.sum()
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return s / total


def _weighted_return_series(returns_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    w = _normalize_weights(weights)
    missing = [k for k in w.index if k not in returns_df.columns]
    if missing:
        raise ValueError(f"Tickers missing from Returns_Annual: {missing}")
    port = (returns_df[w.index] * w).sum(axis=1).dropna()
    port.name = "portfolio_return"
    return port


def _sample_iid(series: Sequence[float], n: int) -> np.ndarray:
    return np.random.choice(np.asarray(series), size=n, replace=True)


def _sample_block(series: Sequence[float], years: int, block_size: int) -> np.ndarray:
    vals = np.asarray(series)
    blocks = []
    while sum(len(b) for b in blocks) < years:
        start = np.random.randint(0, len(vals) - block_size + 1)
        blocks.append(vals[start:start + block_size])
    return np.concatenate(blocks)[:years]


def simulate_portfolio(
    market: dict,
    cfg: PortfolioConfig,
    contributions_by_year: Optional[pd.Series] = None, 
    use_inflation: bool = True,
) -> dict:
    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    annual_rets = market["returns_annual"]
    port_series = _weighted_return_series(annual_rets, cfg.weights).values

    if cfg.sample_mode == "iid":
        draw_returns = lambda n_years: _sample_iid(port_series, n_years)
    elif cfg.sample_mode == "block":
        draw_returns = lambda n_years: _sample_block(port_series, n_years, cfg.block_size)
    else:
        raise ValueError("sample_mode must be 'iid' or 'block'")

    infl_series = market.get("inflation_annual") if use_inflation else None
    if infl_series is not None and len(infl_series) > 0:
        infl_vals = infl_series.dropna().values
        draw_infl = lambda n_years: _sample_iid(infl_vals, n_years)
    else:
        draw_infl = None

    Y, N = cfg.years, cfg.n_sims
    nominal = np.zeros((Y + 1, N), dtype=float)
    nominal[0, :] = cfg.initial_balance
    real = np.zeros_like(nominal) if draw_infl is not None else None
    if real is not None:
        real[0, :] = cfg.initial_balance

    if contributions_by_year is None:
        contributions_by_year = pd.Series(0.0, index=range(Y))
    else:
        contributions_by_year = contributions_by_year.reindex(range(Y)).fillna(0.0)

    if draw_infl is not None:
        cumulative_deflator = np.ones(N, dtype=float)

    for t in range(1, Y + 1):
        contrib = contributions_by_year.iloc[t - 1]
        r_year = draw_returns(N)
        nominal[t, :] = (nominal[t - 1, :] + contrib) * (1.0 + r_year)

        if draw_infl is not None:
            inf_year = draw_infl(N)
            cumulative_deflator = cumulative_deflator / (1.0 + inf_year)
            real[t, :] = nominal[t, :] * cumulative_deflator

    def _pct_df(arr: np.ndarray) -> pd.DataFrame:
        idx = pd.RangeIndex(0, Y + 1, name="Year")
        return pd.DataFrame(
            {
                "p10": np.percentile(arr, 10, axis=1),
                "p50": np.percentile(arr, 50, axis=1),
                "p90": np.percentile(arr, 90, axis=1),
            },
            index=idx,
        )

    out = {
        "nominal_paths": nominal,
        "percentiles_nominal": _pct_df(nominal),
        "real_paths": None,
        "percentiles_real": None,
    }
    if real is not None:
        out["real_paths"] = real
        out["percentiles_real"] = _pct_df(real)
    return out
