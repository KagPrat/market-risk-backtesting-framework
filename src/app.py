import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_loader import fetch_data, STRESS_PERIODS, TICKERS, WEIGHTS
from var_models import compute_all_vars
from backtesting import run_backtest, stress_period_summary
from features import FEATURE_COLS

st.set_page_config(
    page_title="VaR Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_COLORS = {
    "Parametric":     "#3B82F6",
    "Historical Sim": "#10B981",
    "GARCH(1,1)":    "#F59E0B",
    "ML (XGBoost)":  "#EF4444",
}
ZONE_COLORS = {
    "Green":  "#10B981",
    "Yellow": "#F59E0B",
    "Red":    "#EF4444",
}

with st.sidebar:
    st.title("⚙️ Settings")
    confidence = st.slider("Confidence Level", 0.90, 0.99, 0.99, 0.01, format="%.2f")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2005-01-01"))
    end_date   = st.date_input("End Date",   value=pd.to_datetime("2024-12-31"))
    garch_step = st.selectbox(
        "GARCH Re-fit Frequency (days)",
        options=[5, 10, 21, 63],
        index=2,
        help="How often to re-fit GARCH. Lower = slower but more accurate."
    )
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    st.divider()
    st.subheader("Portfolio Weights")
    for i, ticker in enumerate(TICKERS):
        st.write(f"**{ticker}**: {WEIGHTS[i]*100:.0f}%")

st.title("📉 VaR Model Comparison & Backtesting Dashboard")
st.markdown(
    "Compares **Parametric**, **Historical Simulation**, **GARCH(1,1)**, and "
    "**ML (XGBoost)** VaR estimates on a 5-asset portfolio across real market stress periods."
)

@st.cache_data(show_spinner=False)
def load_data(start, end):
    return fetch_data(str(start), str(end))

@st.cache_data(show_spinner=False)
def run_models(port_returns_json, vix_json, confidence, garch_step):
    port_returns = pd.read_json(port_returns_json, typ="series")
    port_returns.index = pd.to_datetime(port_returns.index)
    vix = pd.read_json(vix_json, typ="series")
    vix.index = pd.to_datetime(vix.index)
    return compute_all_vars(port_returns, vix, confidence, garch_step)

if run_btn or "var_df" in st.session_state:

    if run_btn:
        with st.spinner("📡 Fetching market data..."):
            data = load_data(start_date, end_date)
        st.session_state["data"] = data

        with st.spinner("🔬 Fitting models — GARCH + XGBoost may take ~60s..."):
            var_df, ml_model = run_models(
                data["portfolio_returns"].to_json(),
                data["vix"].to_json(),
                confidence,
                garch_step,
            )
        st.session_state["var_df"]     = var_df
        st.session_state["ml_model"]   = ml_model
        st.session_state["confidence"] = confidence

    data      = st.session_state["data"]
    var_df    = st.session_state["var_df"]
    ml_model  = st.session_state["ml_model"]
    port_ret  = data["portfolio_returns"]
    vix       = data["vix"]
    conf_used = st.session_state["confidence"]

    backtest  = run_backtest(port_ret, var_df, conf_used)
    stress_df = stress_period_summary(port_ret, var_df, conf_used)

    stress_colors = [
        "rgba(239,68,68,0.12)",
        "rgba(245,158,11,0.12)",
        "rgba(59,130,246,0.12)",
    ]

    st.divider()
    st.subheader("📊 Model Summary")
    kpi_cols = st.columns(len(var_df.columns))
    for i, model in enumerate(var_df.columns):
        k = backtest[model]["kupiec"]
        b = backtest[model]["basel"]
        zone = b["zone"]
        color = ZONE_COLORS[zone]
        with kpi_cols[i]:
            st.markdown(
                f"""
                <div style="background:#1e2130;border-radius:10px;padding:16px;border-left:4px solid {MODEL_COLORS[model]}">
                  <div style="font-size:13px;color:#9ca3af">{model}</div>
                  <div style="font-size:28px;font-weight:700;color:{MODEL_COLORS[model]}">
                    {k['exception_rate']}%
                  </div>
                  <div style="font-size:12px;color:#6b7280">Breach rate (exp: {k['expected_rate']}%)</div>
                  <div style="margin-top:8px">
                    <span style="background:{color};color:#000;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600">
                      Basel: {zone}
                    </span>
                  </div>
                  <div style="font-size:12px;color:#9ca3af;margin-top:4px">
                    Kupiec p={k['p_value']} {'❌ Rejected' if k['reject_h0'] else '✅ Pass'}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 VaR Over Time",
        "💥 Backtesting",
        "🌡️ Stress Periods",
        "🤖 ML Insights",
    ])

else:
    st.info("👈 Configure settings in the sidebar and click **Run Analysis** to begin.")

with tab1:
        st.subheader("VaR Estimates vs Portfolio Returns")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.6, 0.4], vertical_spacing=0.04)

        fig.add_trace(go.Bar(
            x=port_ret.index,
            y=port_ret.values,
            marker_color=np.where(port_ret.values < 0, "#EF4444", "#10B981"),
            name="Daily Return",
            opacity=0.6,
        ), row=1, col=1)

        for model, color in MODEL_COLORS.items():
            if model in var_df.columns:
                fig.add_trace(go.Scatter(
                    x=var_df.index,
                    y=-var_df[model],
                    line=dict(color=color, width=1.5),
                    name=f"{model} VaR",
                ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=vix.index,
            y=vix.values,
            line=dict(color="#A78BFA", width=1),
            name="VIX",
        ), row=2, col=1)

        for (pname, (s, e)), sc in zip(STRESS_PERIODS.items(), stress_colors):
            for row in [1, 2]:
                fig.add_vrect(
                    x0=s, x1=e, fillcolor=sc, line_width=0,
                    annotation_text=pname if row == 1 else "",
                    annotation_position="top left",
                    annotation_font_size=10,
                    row=row, col=1
                )

        fig.update_layout(
            height=600, template="plotly_dark",
            legend=dict(orientation="h", y=-0.05),
            margin=dict(l=0, r=0, t=20, b=0),
        )
        fig.update_yaxes(title_text="Return", row=1, col=1)
        fig.update_yaxes(title_text="VIX", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("VaR Exceptions Over Time")
        fig2 = go.Figure()

        for model, color in MODEL_COLORS.items():
            if model not in var_df.columns:
                continue
            exc = backtest[model]["exceptions"]
            exc_dates = exc[exc].index
            exc_returns = port_ret.reindex(exc_dates)
            fig2.add_trace(go.Scatter(
                x=exc_dates,
                y=exc_returns.values,
                mode="markers",
                marker=dict(color=color, size=6, symbol="x"),
                name=f"{model} breach",
            ))

        fig2.add_trace(go.Scatter(
            x=port_ret.index,
            y=port_ret.values,
            line=dict(color="rgba(255,255,255,0.2)", width=0.8),
            name="Returns",
            showlegend=False,
        ))

        for (pname, (s, e)), sc in zip(STRESS_PERIODS.items(), stress_colors):
            fig2.add_vrect(
                x0=s, x1=e, fillcolor=sc, line_width=0,
                annotation_text=pname, annotation_position="top left",
                annotation_font_size=10
            )

        fig2.update_layout(height=420, template="plotly_dark",
                           margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Kupiec POF Test Results")
        kupiec_rows = []
        for model in var_df.columns:
            k = backtest[model]["kupiec"]
            b = backtest[model]["basel"]
            kupiec_rows.append({
                "Model": model,
                "Observations": k["n_obs"],
                "Exceptions": k["n_exceptions"],
                "Expected": k["expected_exceptions"],
                "Breach Rate (%)": k["exception_rate"],
                "LR Statistic": k["lr_stat"],
                "p-value": k["p_value"],
                "Reject H₀": "❌ Yes" if k["reject_h0"] else "✅ Pass",
                "Basel Zone": b["zone"],
            })
        st.dataframe(
            pd.DataFrame(kupiec_rows).set_index("Model"),
            use_container_width=True,
        )

    with tab3:
        st.subheader("Exception Rates During Stress Periods")
        fig3 = px.bar(
            stress_df,
            x="Period", y="Exception Rate (%)",
            color="Model", barmode="group",
            color_discrete_map=MODEL_COLORS,
            template="plotly_dark",
            text="Exception Rate (%)",
        )
        fig3.add_hline(
            y=(1 - conf_used) * 100, line_dash="dash",
            line_color="white",
            annotation_text=f"Expected ({(1-conf_used)*100:.1f}%)"
        )
        fig3.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Overshoot Ratio Heatmap")
        pivot = stress_df.pivot(index="Period", columns="Model", values="Overshoot Ratio")
        fig4 = px.imshow(
            pivot, text_auto=True, aspect="auto",
            color_continuous_scale="RdYlGn_r",
            template="plotly_dark",
        )
        fig4.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig4, use_container_width=True)

        st.dataframe(stress_df.set_index(["Period", "Model"]), use_container_width=True)

    with tab4:
        st.subheader("XGBoost Feature Importance")
        importance = pd.Series(
            ml_model.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=True)

        fig5 = go.Figure(go.Bar(
            x=importance.values,
            y=importance.index,
            orientation="h",
            marker_color="#EF4444",
        ))
        fig5.update_layout(
            height=420, template="plotly_dark",
            xaxis_title="Importance Score",
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig5, use_container_width=True)

        st.subheader("ML vs Parametric VaR Difference")
        if "Parametric" in var_df.columns and "ML (XGBoost)" in var_df.columns:
            diff = var_df["ML (XGBoost)"] - var_df["Parametric"]
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(
                x=diff.index, y=diff.values,
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.2)",
                line=dict(color="#EF4444", width=1),
                name="ML − Parametric",
            ))
            for (pname, (s, e)), sc in zip(STRESS_PERIODS.items(), stress_colors):
                fig6.add_vrect(
                    x0=s, x1=e, fillcolor=sc, line_width=0,
                    annotation_text=pname, annotation_position="top left",
                    annotation_font_size=10
                )
            fig6.add_hline(y=0, line_color="white", line_dash="dash")
            fig6.update_layout(height=380, template="plotly_dark",
                               margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig6, use_container_width=True)
            st.caption(
                "Positive = ML predicts higher risk than Parametric. "
                "ML typically spikes during stress due to VIX and realized vol features."
            )