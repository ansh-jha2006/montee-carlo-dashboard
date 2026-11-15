from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bond_math as bm

LOGO = Path(__file__).parent / "logo.png"

PRIMARY = "#4b64ff"
VIOLET = "#7a5cff"
ACCENT = "#ff6b4a"
INK = "#1b1f3b"
MUTED = "#5b6291"
GRID = "#e8ebfa"

st.set_page_config(
    page_title="Bond Analysis Dashboard",
    page_icon=str(LOGO) if LOGO.exists() else None,
    layout="wide",
)

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }}
    #MainMenu, footer {{ visibility: hidden; }}
    .hero {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {VIOLET} 100%);
        border-radius: 18px; padding: 1.7rem 2.1rem; margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(75, 100, 255, 0.22);
    }}
    .hero-title {{ color: #fff; font-size: 2rem; font-weight: 700; line-height: 1.2; }}
    .hero-sub {{ color: rgba(255,255,255,0.88); font-size: 1rem; margin-top: 0.4rem; }}
    .hero-chip {{
        display: inline-block; background: rgba(255,255,255,0.18); color: #fff;
        border-radius: 999px; padding: 0.2rem 0.8rem; margin: 0.8rem 0.4rem 0 0;
        font-size: 0.82rem; font-weight: 600;
    }}
    .section-label {{
        color: {MUTED}; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; margin: 1.2rem 0 0.5rem;
    }}
    [data-testid="stMetric"] {{
        background: #fff; border: 1px solid {GRID}; border-radius: 14px;
        padding: 14px 18px; box-shadow: 0 1px 4px rgba(75, 100, 255, 0.08);
    }}
    [data-testid="stMetricLabel"] {{ color: {MUTED}; }}
    [data-testid="stMetricValue"] {{ color: {INK}; font-weight: 700; }}
    button[data-baseweb="tab"] {{ font-weight: 600; }}
    [data-testid="stSidebar"] {{ border-right: 1px solid {GRID}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def money(x: float) -> str:
    return f"₹{x:,.2f}"


def style_fig(fig: go.Figure, title: str, xtitle: str, ytitle: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=INK)),
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(color=INK),
        xaxis=dict(title=xtitle, gridcolor=GRID, zeroline=False),
        yaxis=dict(title=ytitle, gridcolor=GRID, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white"),
    )
    return fig


@st.cache_data(show_spinner=False)
def run_simulation(face_value, coupon_rate, years, ytm, iterations, yield_vol, freq, seed):
    prices = bm.monte_carlo_prices(face_value, coupon_rate, years, ytm, iterations,
                                   yield_vol, freq, seed)
    return prices, bm.summarize_prices(prices)


# ---- Sidebar inputs -----------------------------------------------------------
st.sidebar.markdown("### Bond Parameters")
face_value = st.sidebar.number_input("Face Value (₹)", min_value=0.01, value=1000.00, format="%.2f")
coupon_pct = st.sidebar.number_input("Coupon Rate (%)", min_value=0.0, max_value=100.0, value=5.00, format="%.2f")
freq = st.sidebar.selectbox("Coupon Payments per Year", [1, 2, 4, 12], index=1,
                            format_func=lambda f: {1: "1 (annual)", 2: "2 (semiannual)",
                                                   4: "4 (quarterly)", 12: "12 (monthly)"}[f])
years = st.sidebar.number_input("Years to Maturity", min_value=1.0 / freq, max_value=100.0,
                                value=10.00, step=0.5, format="%.2f")
current_price = st.sidebar.number_input("Current Bond Price (₹)", min_value=0.01, value=950.00, format="%.2f")

st.sidebar.divider()
st.sidebar.markdown("### Scenario & Simulation")
interest_shock = st.sidebar.slider("Interest Rate Shock (%)", -3.0, 3.0, 0.0, 0.1,
                                   help="Parallel shift applied to the yield to maturity.")
iterations = st.sidebar.slider("Monte Carlo Simulation Count", 5, 10000, 1000, 5)
yield_vol_pct = st.sidebar.slider("Yield Volatility, σ (%)", 0.1, 5.0, 1.0, 0.1,
                                  help="Standard deviation of simulated yields, in percentage points.")
seed = st.sidebar.number_input("Random Seed", min_value=0, value=42, step=1,
                               help="Same seed gives the same simulation. Change it for a new sample.")

coupon_rate = coupon_pct / 100.0
yield_vol = yield_vol_pct / 100.0

# ---- Header -------------------------------------------------------------------
freq_name = {1: "annual", 2: "semiannual", 4: "quarterly", 12: "monthly"}[freq]
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-title">Bond Analysis Dashboard</div>
      <div class="hero-sub">Yield, duration, convexity, rate shocks and Monte Carlo price risk for a fixed-income bond.</div>
      <span class="hero-chip">Face {money(face_value)}</span>
      <span class="hero-chip">Coupon {coupon_pct:.2f}% {freq_name}</span>
      <span class="hero-chip">{years:g} yrs to maturity</span>
      <span class="hero-chip">Price {money(current_price)}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Core analytics -------------------------------------------------------------
ytm = bm.yield_to_maturity(face_value, coupon_rate, years, current_price, freq)
if ytm is None:
    st.error("Could not solve for a yield to maturity with these inputs. "
             "Check that the price, coupon, and maturity are realistic.")
    st.stop()

model_price = bm.bond_price(face_value, coupon_rate, years, ytm, freq)
mac_duration = bm.macaulay_duration(face_value, coupon_rate, years, ytm, freq)
mod_duration = bm.modified_duration(face_value, coupon_rate, years, ytm, freq)
bond_convexity = bm.convexity(face_value, coupon_rate, years, ytm, freq)
bond_yield = bm.current_yield(face_value, coupon_rate, current_price)

shocked_ytm = ytm + interest_shock / 100.0
if shocked_ytm <= bm.MIN_YTM:
    st.error("The interest rate shock pushes the yield below the model's valid range.")
    st.stop()
price_shocked = bm.bond_price(face_value, coupon_rate, years, shocked_ytm, freq)
shock_change = price_shocked - current_price
if abs(shock_change) < 1e-9:  # avoid displaying float noise as "-0.00"
    shock_change = 0.0
shock_pct = shock_change / current_price * 100.0
dy = interest_shock / 100.0
approx_pct = (-mod_duration * dy + 0.5 * bond_convexity * dy**2) * 100.0  # duration + convexity estimate

par_pct = (current_price / face_value - 1.0) * 100.0
if abs(par_pct) < 0.005:
    status = "At par"
elif par_pct < 0:
    status = f"Discount · {par_pct:+.2f}% vs par"
else:
    status = f"Premium · {par_pct:+.2f}% vs par"

# ---- KPI cards ------------------------------------------------------------------
st.markdown('<div class="section-label">Valuation & risk</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Yield to Maturity", f"{ytm * 100:.4f}%", help="Solved from the market price; compounded at the coupon frequency.")
k2.metric("Current Yield", f"{bond_yield * 100:.2f}%", help="Annual coupon divided by market price.")
k3.metric("Market Price", money(current_price), status, delta_color="off")
k4.metric("Price at YTM (check)", money(model_price),
          help="Repricing the bond at the solved YTM. It should equal the market price.")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Macaulay Duration", f"{mac_duration:.2f} yrs", help="Weighted-average time to receive the cash flows.")
k6.metric("Modified Duration", f"{mod_duration:.2f}", help="Approximate % price change per 1% change in yield.")
k7.metric("Convexity", f"{bond_convexity:.2f}", help="Curvature of the price/yield relationship, in years squared.")
if interest_shock == 0:
    k8.metric("Price after Shock", money(price_shocked), "no shock applied", delta_color="off")
else:
    k8.metric(f"Price after {interest_shock:+.1f}% Shock", money(price_shocked),
              f"{shock_change:+,.2f} ({shock_pct:+.2f}%)",
              help=f"Duration + convexity estimate of the change: {approx_pct:+.2f}%")

# ---- Charts & tables --------------------------------------------------------------
tab_curve, tab_mc, tab_cf, tab_info = st.tabs(
    ["📈 Price vs Yield", "🎲 Monte Carlo", "💵 Cash Flows", "ℹ️ Methodology"])

with tab_curve:
    curve_ytms = np.linspace(max(0.001, ytm - 0.05), ytm + 0.05, 80)
    curve_prices = bm.bond_price(face_value, coupon_rate, years, curve_ytms, freq)
    tangent = current_price * (1.0 - mod_duration * (curve_ytms - ytm))

    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=curve_ytms * 100, y=curve_prices, mode="lines", name="Actual price",
        line=dict(color=PRIMARY, width=3), fill="tozeroy", fillcolor="rgba(75,100,255,0.06)",
        hovertemplate="YTM %{x:.2f}%<br>Price ₹%{y:,.2f}<extra></extra>"))
    fig_curve.add_trace(go.Scatter(
        x=curve_ytms * 100, y=tangent, mode="lines", name="Duration estimate (linear)",
        line=dict(color=MUTED, width=1.5, dash="dot"),
        hovertemplate="YTM %{x:.2f}%<br>Linear est. ₹%{y:,.2f}<extra></extra>"))
    fig_curve.add_trace(go.Scatter(
        x=[ytm * 100], y=[current_price], mode="markers", name="Current",
        marker=dict(size=12, color=ACCENT, line=dict(color="white", width=2)),
        hovertemplate="Current<br>YTM %{x:.3f}%<br>Price ₹%{y:,.2f}<extra></extra>"))
    if interest_shock != 0:
        fig_curve.add_trace(go.Scatter(
            x=[shocked_ytm * 100], y=[price_shocked], mode="markers", name="After shock",
            marker=dict(size=12, color=VIOLET, symbol="diamond", line=dict(color="white", width=2)),
            hovertemplate="After shock<br>YTM %{x:.3f}%<br>Price ₹%{y:,.2f}<extra></extra>"))
    style_fig(fig_curve, "Price vs Yield", "Yield to Maturity (%)", "Bond Price (₹)", height=460)
    fig_curve.update_yaxes(range=[max(0, min(curve_prices.min(), tangent.min()) * 0.95),
                                  max(curve_prices.max(), tangent.max()) * 1.03])
    st.plotly_chart(fig_curve)
    st.caption("The gap between the solid curve and the dotted line is convexity: the actual price "
               "rises more than duration predicts when yields fall, and falls less when they rise.")

with tab_mc:
    prices, mc = run_simulation(face_value, coupon_rate, years, ytm, iterations,
                                yield_vol, freq, int(seed))
    p_below = float(np.mean(prices < current_price)) * 100.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean Price", money(mc["mean_price"]))
    m2.metric("Volatility (Std Dev)", money(mc["std_dev"]))
    m3.metric("5th Percentile", money(mc["p5"]), help="Lower bound of the central 90% of outcomes.")
    m4.metric("95th Percentile", money(mc["p95"]), help="Upper bound of the central 90% of outcomes.")
    m5, m6, m7, _ = st.columns(4)
    m5.metric("Min Simulated", money(mc["min_price"]))
    m6.metric("Max Simulated", money(mc["max_price"]))
    m7.metric("P(Price < Market)", f"{p_below:.1f}%")

    fig_mc = go.Figure()
    fig_mc.add_trace(go.Histogram(
        x=prices, nbinsx=min(100, max(10, iterations // 10)), marker=dict(color=PRIMARY),
        opacity=0.85, name="Simulated prices",
        hovertemplate="Price ₹%{x:,.2f}<br>Count %{y}<extra></extra>"))
    fig_mc.add_vrect(x0=mc["p5"], x1=mc["p95"], fillcolor=VIOLET, opacity=0.08, line_width=0)
    fig_mc.add_vline(x=current_price, line_dash="dash", line_color=ACCENT,
                     annotation_text="Market price", annotation_position="top left")
    fig_mc.add_vline(x=mc["mean_price"], line_color=INK, line_width=1.5,
                     annotation_text="Mean", annotation_position="top right")
    style_fig(fig_mc, f"Simulated Bond Price Distribution ({iterations:,} iterations, σ = {yield_vol_pct:.1f}%)",
              "Bond Price (₹)", "Frequency", height=440)
    fig_mc.update_layout(showlegend=False)
    st.plotly_chart(fig_mc)
    st.caption("Shaded band: central 90% of outcomes (5th to 95th percentile).")

with tab_cf:
    times, flows = bm.cash_flows(face_value, coupon_rate, years, freq)
    discount = (1.0 + ytm / freq) ** (freq * times)
    pv = flows / discount
    cf_df = pd.DataFrame({
        "Period": np.arange(1, len(times) + 1),
        "Year": times,
        "Cash Flow (₹)": flows,
        "Discount Factor": 1.0 / discount,
        "Present Value (₹)": pv,
        "% of Price": [f"{v:.2f}%" for v in pv / pv.sum() * 100.0],
    })

    fig_cf = go.Figure()
    fig_cf.add_trace(go.Bar(x=cf_df["Year"], y=cf_df["Present Value (₹)"], marker=dict(color=PRIMARY),
                            hovertemplate="Year %{x:.2f}<br>PV ₹%{y:,.2f}<extra></extra>"))
    style_fig(fig_cf, "Present Value of Each Cash Flow", "Years from now", "Present Value (₹)", height=380)
    fig_cf.update_layout(showlegend=False)
    st.plotly_chart(fig_cf)
    st.dataframe(
        cf_df, hide_index=True,
        column_config={
            "Year": st.column_config.NumberColumn(format="%.2f"),
            "Cash Flow (₹)": st.column_config.NumberColumn(format="₹%.2f"),
            "Discount Factor": st.column_config.NumberColumn(format="%.4f"),
            "Present Value (₹)": st.column_config.NumberColumn(format="₹%.2f"),
        })
    st.caption("The final period includes the return of face value. Present values sum to the price at the solved YTM.")

with tab_info:
    st.markdown(
        """
**How the numbers are calculated**

- **Yield to maturity** is solved numerically (Brent's method) so that the discounted cash flows equal the market price. It is a nominal annual rate compounded at the coupon frequency.
- **Duration and convexity** use the same cash flows and compounding as pricing. Macaulay duration is in years; modified duration is Macaulay divided by (1 + YTM / frequency).
- **Rate shock** reprices the bond exactly at YTM + shock. The duration + convexity estimate in the card's tooltip is a second-order approximation for comparison.
- **Monte Carlo** draws yields from a normal distribution centred on the YTM with the chosen volatility, then reprices the bond at each draw. Simulated yields are floored at -99%.

**Limits of the model**

- Yield changes are independent, normally distributed and unrelated to the bond's maturity: there is no mean reversion, term structure or credit risk.
- A mean simulated price slightly above the market price is expected, because convexity makes price gains from falling yields larger than losses from rising yields.
- This is an educational tool, not investment advice.
        """
    )
