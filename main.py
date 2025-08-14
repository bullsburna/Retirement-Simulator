from pathlib import Path
import os
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

FRED_CPI_SERIES = "CPIAUCSL"   # CPI (All Urban Consumers, SA)
FRED_API_KEY = os.getenv("FRED_API_KEY", "")


# ----------------------------
# Helpers
# ----------------------------
def get_monthly_adjusted_prices(ticker: str, start=START, end=END) -> pd.Series:
    """
    Use auto_adjust=True so 'Close' is dividend/split adjusted (total-return proxy).
    Download monthly bars, resample to month-end to ensure stable index.
    """
    df = yf.download(ticker, start=start, end=end,
                     interval="1mo", auto_adjust=True,
                     progress=False, threads=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. "
                         f"Check ticker/date range/internet.")
    s = df["Close"].copy().resample("M").last().dropna()
    s.name = ticker
    return s


def concat_align(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.concat(series_map.values(), axis=1)
    # Keep overlapping region; drop months where all tickers are NaN
    df = df.dropna(how="all")
    return df


def annual_returns_from_monthly_prices(prices_m: pd.DataFrame) -> pd.DataFrame:
    annual_px = prices_m.resample("Y").last()
    rets = annual_px.pct_change().dropna()
    rets.index = rets.index.year
    return rets


def fetch_cpi_monthly(api_key: str) -> pd.DataFrame:
    """
    Fetch CPI from FRED and compute monthly YoY inflation (12-mo change).
    Requires 'fredapi' to be installed and FRED_API_KEY set.
    """
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set. Set it to fetch CPI.")
    try:
        from fredapi import Fred
    except ImportError as e:
        raise ImportError("fredapi not installed. Run: pip install fredapi") from e

    fred = Fred(api_key=api_key)
    cpi = fred.get_series(FRED_CPI_SERIES)  # pandas Series with DatetimeIndex
    df = cpi.to_frame("CPI").asfreq("M")
    df["inflation_yoy"] = df["CPI"].pct_change(12)
    return df


def annual_cpi_with_yoy(cpi_m: pd.DataFrame) -> pd.DataFrame:
    annual = cpi_m.resample("Y").mean()
    annual.index = annual.index.year
    annual["inflation_yoy"] = annual["CPI"].pct_change()
    return annual


# ----------------------------
# Main
# ----------------------------
def main():
    # ETFs: monthly adjusted prices
    price_map: dict[str, pd.Series] = {}
    for t in TICKERS:
        price_map[t] = get_monthly_adjusted_prices(t)

    prices_m = concat_align(price_map)                 # Prices_Monthly
    returns_m = prices_m.pct_change().dropna()         # Returns_Monthly
    returns_y = annual_returns_from_monthly_prices(prices_m)  # Returns_Annual

    # CPI (optional)
    cpi_monthly = cpi_annual = None
    if FRED_API_KEY:
        try:
            cpi_monthly = fetch_cpi_monthly(FRED_API_KEY)
            cpi_annual = annual_cpi_with_yoy(cpi_monthly)
        except Exception as e:
            print(f"WARNING: CPI fetch failed ({e}). Skipping CPI sheets.")
    else:
        print("NOTE: FRED_API_KEY not set — skipping CPI sheets.")

    # Write to one Excel workbook
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as xw:
        prices_m.to_excel(xw, sheet_name="Prices_Monthly")
        returns_m.to_excel(xw, sheet_name="Returns_Monthly")
        returns_y.to_excel(xw, sheet_name="Returns_Annual")
        if cpi_monthly is not None:
            cpi_monthly.dropna().to_excel(xw, sheet_name="CPI_Monthly")
        if cpi_annual is not None:
            cpi_annual.dropna().to_excel(xw, sheet_name="CPI_Annual")

    print(f"Wrote: {OUT_XLSX.resolve()}")
    print("Sheets: Prices_Monthly, Returns_Monthly, Returns_Annual,"
          " CPI_Monthly (opt), CPI_Annual (opt)")


if __name__ == "__main__":
    main()
