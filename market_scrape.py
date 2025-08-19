from pathlib import Path
import os
from typing import Dict
import pandas as pd
import yfinance as yf

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OUT_XLSX = DATA_DIR / "market_inputs.xlsx"

TICKERS = [
    "VOO",   # US equity (S&P 500)
    "BND",   # US aggregate bonds
    "VXUS",  # Intl equity ex-US
    "VNQ",   # REITs
    "SCHD",  # Dividend equity
]

START = "2010-01-01" 
END = None          

FRED_CPI_SERIES = "CPIAUCSL"
FRED_API_KEY = os.getenv("FRED_API_KEY", "")


def get_monthly_adjusted_prices(ticker: str, start: str = START, end: str | None = END) -> pd.Series:
    """
    Use auto_adjust=True so 'Close' is dividend/split-adjusted (total-return proxy).
    Download monthly bars and enforce month-end index.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker/date range/internet.")
    s = df["Close"].copy().resample("ME").last().dropna()
    s.name = ticker
    return s


def concat_align(series_map: Dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.concat(series_map.values(), axis=1)
    return df.dropna(how="all")


def align_to_common_start(prices_m: pd.DataFrame) -> pd.DataFrame:
    """
    Trim to the first date where ALL columns have data (no leading NaNs).
    Prints the chosen common start date.
    """
    first_valid = {col: prices_m[col].first_valid_index() for col in prices_m.columns}
    common_start = max(d for d in first_valid.values() if d is not None)
    trimmed = prices_m.loc[prices_m.index >= common_start].copy()
    trimmed = trimmed.dropna(how="any")
    print(f"Clean common start set to: {common_start.date()}")
    return trimmed


def annual_returns_from_monthly_prices(prices_m: pd.DataFrame) -> pd.DataFrame:
    annual_px = prices_m.resample("YE").last()
    rets = annual_px.pct_change().dropna(how="all")
    rets.index = rets.index.year
    return rets


def fetch_cpi_monthly(api_key: str) -> pd.DataFrame:
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set. Set it to fetch CPI.")
    try:
        from fredapi import Fred
    except ImportError as e:
        raise ImportError("fredapi not installed. Run: pip install fredapi") from e

    fred = Fred(api_key=api_key)
    s = fred.get_series(FRED_CPI_SERIES)
    if s is None or len(s) == 0:
        raise ValueError("FRED returned empty CPI series")

    df = s.to_frame(name="CPI").asfreq("MS")
    df["inflation_yoy"] = df["CPI"].pct_change(12)
    return df


def annual_cpi_with_yoy(cpi_m: pd.DataFrame) -> pd.DataFrame:
    annual = cpi_m.resample("YE").mean()
    annual.index = annual.index.year
    annual["inflation_yoy"] = annual["CPI"].pct_change()
    return annual

def main() -> None:

    price_map: Dict[str, pd.Series] = {}
    for t in TICKERS:
        try:
            price_map[t] = get_monthly_adjusted_prices(t)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch prices for {t}: {e}") from e

    prices_m_raw = concat_align(price_map)
    prices_m = align_to_common_start(prices_m_raw)      
    returns_m = prices_m.pct_change().dropna(how="all")  
    returns_y = annual_returns_from_monthly_prices(prices_m)  

    cpi_monthly = cpi_annual = None
    if FRED_API_KEY:
        try:
            cpi_monthly = fetch_cpi_monthly(FRED_API_KEY)
            if cpi_monthly.empty:
                raise ValueError("CPI returned empty DataFrame")
            cpi_annual = annual_cpi_with_yoy(cpi_monthly)
            if cpi_annual.empty:
                raise ValueError("Annual CPI returned empty DataFrame")
            print("CPI fetched successfully.")
        except Exception as e:
            print(f"WARNING: CPI fetch failed ({e}). Skipping CPI sheets.")
            cpi_monthly = None
            cpi_annual = None
    else:
        print("NOTE: FRED_API_KEY not set — skipping CPI sheets.")

    def write_excel(path: Path) -> None:
        with pd.ExcelWriter(path, engine="xlsxwriter", datetime_format="yyyy-MM-dd") as xw:
            prices_m.to_excel(xw, sheet_name="Prices_Monthly")
            returns_m.to_excel(xw, sheet_name="Returns_Monthly")
            returns_y.to_excel(xw, sheet_name="Returns_Annual")
            if cpi_monthly is not None:
                cpi_monthly.dropna(how="all").to_excel(xw, sheet_name="CPI_Monthly")
            if cpi_annual is not None:
                cpi_annual.dropna(how="all").to_excel(xw, sheet_name="CPI_Annual")

    try:
        write_excel(OUT_XLSX)
        final_path = OUT_XLSX
    except PermissionError:
        fallback = DATA_DIR / f"market_inputs_{pd.Timestamp.now():%Y%m%d_%H%M%S}.xlsx"
        write_excel(fallback)
        print(f"Existing file appears open/locked. Wrote to: {fallback}")
        final_path = fallback

    print(f"Wrote: {final_path.resolve()}")
    print(
        "Sheets: Prices_Monthly, Returns_Monthly, Returns_Annual,"
        " CPI_Monthly (opt), CPI_Annual (opt)"
    )


if __name__ == "__main__":
    main()
