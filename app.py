import streamlit as st
import asyncio
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
import numpy as np

from scraper.crawler import TacticalParser
from scraper.prediction import PredictionModel

try:
    import nest_asyncio

    nest_asyncio.apply()
except Exception:
    pass

st.set_page_config(
    page_title="TDI-Engine Dashboard", layout="wide", page_icon="⚽"
)

st.title("⚽ TDI-Engine: Football Tactical Dashboard")
st.markdown(
    "Analyze real match data from FotMob and predict match momentum/tactical clash."
)

col1, col2, col3 = st.columns(3)
with col1:
    team1_name = st.text_input("Team 1 Name", value="Arsenal")
with col2:
    team2_name = st.text_input("Team 2 Name", value="Real Madrid")
with col3:
    tournament = st.text_input(
        "Tournament (Optional)", value="Champions League"
    )

if st.button("Run Analysis 🚀"):
    st.info(f"Fetching real data for {team1_name} and {team2_name}...")

    async def run_pipeline():
        with st.spinner(
            "Extracting recent match facts & parsing tactical metrics..."
        ):
            t1_base, t2_base = await asyncio.gather(
                TacticalParser.get_aggregated_team_tactics(team1_name),
                TacticalParser.get_aggregated_team_tactics(team2_name),
            )
            return t1_base, t2_base

    t1_base, t2_base = asyncio.run(run_pipeline())

    if not t1_base or not t2_base:
        st.error("Failed to fetch data. Check team names or try again later.")
    else:
        st.success("Data extraction and parsing complete!")

        t1_pred, t2_pred = PredictionModel.generate_prediction(t1_base, t2_base)
        probs = PredictionModel.bivariate_poisson_mc(t1_pred, t2_pred)

        st.subheader("Match Prediction & Analytics")

        col_chart1, col_chart2 = st.columns([2, 1])

        with col_chart1:
            st.markdown("### Momentum & Dominance Over 90 Mins")
            max_min = max(
                100,
                max(
                    np.max(np.nonzero(t1_pred.momentum)),
                    np.max(np.nonzero(t2_pred.momentum)),
                )
                + 5,
            )
            minutes = np.arange(max_min)
            net_momentum = (
                t1_pred.momentum[:max_min] - t2_pred.momentum[:max_min]
            )
            smoothed_net = gaussian_filter1d(net_momentum, sigma=1.5)

            fig_mom = go.Figure()
            pos_net = np.where(smoothed_net > 0, smoothed_net, 0)
            fig_mom.add_trace(
                go.Scatter(
                    x=minutes,
                    y=pos_net,
                    fill="tozeroy",
                    mode="lines",
                    line_color="#00e5ff",
                    name=t1_pred.name,
                )
            )

            neg_net = np.where(smoothed_net < 0, smoothed_net, 0)
            fig_mom.add_trace(
                go.Scatter(
                    x=minutes,
                    y=neg_net,
                    fill="tozeroy",
                    mode="lines",
                    line_color="#ff1744",
                    name=t2_pred.name,
                )
            )

            fig_mom.update_layout(
                xaxis_title="Minutes",
                yaxis_title="Momentum",
                template="plotly_dark",
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_mom, use_container_width=True)

            st.markdown("### xG & Key Metrics Comparison")
            metrics_keys = list(t1_pred.metrics.keys())

            fig_bar = go.Figure()
            fig_bar.add_trace(
                go.Bar(
                    y=metrics_keys,
                    x=[-t1_pred.metrics[k] for k in metrics_keys],
                    name=t1_pred.name,
                    orientation="h",
                    marker_color="#00e5ff",
                )
            )
            fig_bar.add_trace(
                go.Bar(
                    y=metrics_keys,
                    x=[t2_pred.metrics[k] for k in metrics_keys],
                    name=t2_pred.name,
                    orientation="h",
                    marker_color="#ff1744",
                )
            )
            fig_bar.update_layout(
                barmode="relative",
                template="plotly_dark",
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis=dict(
                    ticktext=[str(abs(x)) for x in [-1000, -500, 0, 500, 1000]],
                    tickvals=[-1000, -500, 0, 500, 1000],
                ),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_chart2:
            st.markdown("### Match Outcome Probability")
            st.caption("Bivariate Poisson & Monte Carlo Simulation")
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=list(probs.keys()),
                        values=list(probs.values()),
                        hole=0.4,
                        marker_colors=["#00e5ff", "#ff1744", "#888888"],
                    )
                ]
            )
            fig_pie.update_layout(
                template="plotly_dark",
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("### Projected Stats")
            st.metric(
                f"{t1_pred.name} Projected xG",
                f"{t1_pred.metrics['Expected Goals (xG)']:.2f}",
            )
            st.metric(
                f"{t2_pred.name} Projected xG",
                f"{t2_pred.metrics['Expected Goals (xG)']:.2f}",
            )
