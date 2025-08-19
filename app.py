
from pathlib import Path
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from Sim import PortfolioConfig, load_market_inputs, simulate_portfolio


st.set_page_config(page_title="Retirement Monte Carlo", layout="wide")
st.title("Retirement Monte Carlo Simulator")

DATA_PATH = Path("data") / "market_inputs.xlsx"

@st.cache_data(show_spinner=False)
def _load_market(path: Path):
    return load_market_inputs(path)

try:
    market = _load_market(DATA_PATH)
except Exception as e:
    st.error(f"Could not load {DATA_PATH}. Error: {e}")
    st.stop()

returns_annual = market["returns_annual"]
available_tickers = list(returns_annual.columns)

with st.sidebar:
    st.header("Portfolio & Simulation Settings")
    years = st.number_input("Years to simulate", min_value=1, max_value=80, value=35)
    n_sims = st.number_input("Number of simulations", min_value=100, max_value=100000, value=10000, step=100)
    initial_balance = st.number_input("Initial balance ($)", min_value=0.0, value=0.0, step=1000.0, format="%.2f")

    sample_mode = st.selectbox("Return sampling", ["iid", "block"], index=0)
    block_size = st.slider("Block size (years, for block bootstrap)", min_value=2, max_value=10, value=3)
    seed = st.number_input("Random seed (set -1 for random)", value=42)

    use_inflation = st.checkbox("Inflation-adjusted (real dollars)", value=True)

    st.markdown("---")
    st.caption("Pick tickers to include and set weights (they will be normalized).")
    tickers_selected = st.multiselect("Assets", options=available_tickers, default=["VOO", "BND"])
    weights_inputs = {}
    for t in tickers_selected:
        weights_inputs[t] = st.number_input(f"Weight for {t}", min_value=0.0, value=0.5 if t in ["VOO","BND"] else 0.0, step=0.05)

if len(weights_inputs) == 0:
    st.warning("Select at least one asset.")
    st.stop()
weights_series = pd.Series(weights_inputs, dtype=float)
if weights_series.sum() <= 0:
    st.warning("Weights must sum to a positive number.")
    st.stop()
weights = (weights_series / weights_series.sum()).to_dict()

st.subheader("Contribution Schedule")
st.caption("Edit annual contributions at the **start** of each year (row = year index starting at 0).")

default_years = list(range(years))
default_contrib = [5000.0] * min(years, 5) + [10000.0] * max(0, years - 5)

contrib_df = pd.DataFrame({
    "year_idx": default_years,
    "contribution": default_contrib[:years],
})

edited = st.data_editor(
    contrib_df,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "year_idx": st.column_config.NumberColumn("Year (0-based)", step=1, disabled=True),
        "contribution": st.column_config.NumberColumn("Contribution ($)", step=500.0, format="%.2f"),
    },
)

contributions_series = (
    edited.set_index("year_idx")["contribution"]
          .reindex(range(years))
          .fillna(0.0)
)

run = st.button("Run Simulation", type="primary")
if run:
    cfg = PortfolioConfig(
        weights=weights,
        years=years,
        n_sims=int(n_sims),
        initial_balance=float(initial_balance),
        sample_mode=sample_mode,
        block_size=int(block_size),
        seed=(None if seed == -1 else int(seed)),
    )

    try:
        res = simulate_portfolio(
            market=market,
            cfg=cfg,
            contributions_by_year=contributions_series,
            use_inflation=use_inflation,
        )
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        st.stop()

    def plot_percentiles(df: pd.DataFrame, title: str):
        fig = go.Figure()
        x = df.index

        fig.add_trace(go.Scatter(x=x, y=df["p50"], mode="lines", name="Median (p50)"))
        fig.add_trace(go.Scatter(x=x, y=df["p90"], mode="lines", name="p90", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=x, y=df["p10"], mode="lines", name="p10", line=dict(dash="dash")))
        fig.add_traces([
            go.Scatter(
                x=list(x)+list(x[::-1]),
                y=list(df["p90"])+list(df["p10"][::-1]),
                fill="toself",
                fillcolor="rgba(0,0,0,0.05)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        ])
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            template="plotly_white",
        )
        return fig

    left, right = st.columns(2)
    left.plotly_chart(plot_percentiles(res["percentiles_nominal"], "Nominal Portfolio Value"), use_container_width=True)

    if res["percentiles_real"] is not None:
        right.plotly_chart(plot_percentiles(res["percentiles_real"], "Real (Inflation‑Adjusted) Portfolio Value"), use_container_width=True)
    else:
        right.info("Inflation adjustment is off or CPI data missing.")


    st.subheader("Download Results")
    def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=True).encode("utf-8")

    st.download_button(
        "Download nominal percentiles (CSV)",
        data=df_to_csv_bytes(res["percentiles_nominal"]),
        file_name="mc_percentiles_nominal.csv",
        mime="text/csv",
    )
    if res["percentiles_real"] is not None:
        st.download_button(
            "Download real percentiles (CSV)",
            data=df_to_csv_bytes(res["percentiles_real"]),
            file_name="mc_percentiles_real.csv",
            mime="text/csv",
        )
    st.write("**Nominal percentiles (head):**")
    st.dataframe(res["percentiles_nominal"].head(12))
    if res["percentiles_real"] is not None:
        st.write("**Real percentiles (head):**")
        st.dataframe(res["percentiles_real"].head(12))
else:
    st.info("Set your inputs, then click **Run Simulation**.")
