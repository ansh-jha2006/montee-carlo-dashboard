from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import bond_math as bm

LOGO = Path(__file__).parent / "logo.png"

st.set_page_config(
    page_title="Bond Analysis Dashboard",
    page_icon=str(LOGO) if LOGO.exists() else None,
    layout="wide",
)


@st.cache_data(show_spinner=False)
def run_simulation(face_value, coupon_rate, years, ytm, iterations, yield_vol, freq, seed):
    prices = bm.monte_carlo_prices(face_value, coupon_rate, years, ytm, iterations,
                                   yield_vol, freq, seed)
    return prices, bm.summarize_prices(prices)


st.title("Bond Analysis Dashboard")

# ---- Inputs ---------------------------------------------------------------
st.sidebar.header("Bond Parameters")
face_value = st.sidebar.number_input("Face Value ($)", min_value=0.01, value=1000.00, format="%.2f")
coupon_pct = st.sidebar.number_input("Coupon Rate (%)", min_value=0.0, max_value=100.0, value=5.00, format="%.2f")
freq = st.sidebar.selectbox("Coupon Payments per Year", [1, 2, 4, 12], index=1)
years = st.sidebar.number_input("Years to Maturity", min_value=1.0 / freq, max_value=100.0,
                                value=10.00, step=0.5, format="%.2f")
current_price = st.sidebar.number_input("Current Bond Price ($)", min_value=0.01, value=950.00, format="%.2f")

st.sidebar.header("Scenario & Simulation")
interest_shock = st.sidebar.slider("Interest Rate Shock (%)", -3.0, 3.0, 0.0, 0.1)
iterations = st.sidebar.slider("Monte Carlo Simulation Count", 5, 10000, 1000, 5)
yield_vol_pct = st.sidebar.slider("Yield Volatility, σ (%)", 0.1, 5.0, 1.0, 0.1,
                                  help="Standard deviation of simulated yields, in percentage points.")
seed = st.sidebar.number_input("Random Seed", min_value=0, value=42, step=1,
                               help="Same seed gives the same simulation. Change it for a new sample.")

coupon_rate = coupon_pct / 100.0
yield_vol = yield_vol_pct / 100.0

# ---- Core analytics ---------------------------------------------------------
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
dy = interest_shock / 100.0
approx_pct = -mod_duration * dy + 0.5 * bond_convexity * dy**2  # duration + convexity estimate

# ---- Price / yield curve ----------------------------------------------------
curve_lo = max(0.001, ytm - 0.05)
curve_ytms = np.linspace(curve_lo, ytm + 0.05, 60)
curve_prices = bm.bond_price(face_value, coupon_rate, years, curve_ytms, freq)

fig_curve = go.Figure()
fig_curve.add_trace(go.Scatter(x=curve_ytms * 100, y=curve_prices, mode="lines", name="Price vs YTM"))
fig_curve.add_trace(go.Scatter(x=[ytm * 100], y=[current_price], mode="markers",
                               marker=dict(size=10, color="#e4572e"), name="Current"))
fig_curve.update_layout(title="Price vs Yield Curve", xaxis_title="Yield to Maturity (%)",
                        yaxis_title="Bond Price ($)")

# ---- Layout ----------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_curve)
    st.write(f"Impact of {interest_shock:+.1f}% interest rate shock: `${price_shocked:,.2f}` "
             f"(`{shock_change:+,.2f}`, `{shock_change / current_price * 100:+.2f}%`)")
    st.write(f"Duration + convexity estimate of that change: `{approx_pct * 100:+.2f}%`")
    st.write(f"Yield to Maturity (YTM): `{ytm * 100:.4f}%`")
    st.write(f"Current Yield: `{bond_yield * 100:.2f}%`")
    st.write(f"Bond Price (calculated from YTM): `${model_price:,.2f}`")
    st.write(f"Macaulay Duration: `{mac_duration:.2f}` years")
    st.write(f"Modified Duration: `{mod_duration:.2f}`")
    st.write(f"Bond Convexity: `{bond_convexity:.2f}`")

with col2:
    prices, mc_stats = run_simulation(face_value, coupon_rate, years, ytm, iterations,
                                      yield_vol, freq, int(seed))
    fig = px.histogram(x=prices, nbins=min(100, max(10, iterations // 10)),
                       title=f"Monte Carlo Simulation of Bond Price ({iterations:,} iterations)",
                       color_discrete_sequence=["#4b64ff"])
    fig.add_vline(x=current_price, line_dash="dash", annotation_text="Market price")
    fig.update_layout(xaxis_title="Bond Price", yaxis_title="Frequency", showlegend=False)
    st.plotly_chart(fig)
    st.write(f"Monte Carlo Simulation Statistics ({iterations:,} iterations, σ = {yield_vol_pct:.1f}%)")
    st.write(f"Mean Price: `${mc_stats['mean_price']:,.2f}`")
    st.write(f"Standard Deviation (Volatility): `${mc_stats['std_dev']:,.2f}`")
    st.write(f"5th Percentile (Lower Bound): `${mc_stats['p5']:,.2f}`")
    st.write(f"95th Percentile (Upper Bound): `${mc_stats['p95']:,.2f}`")
    st.write(f"Min Simulated Price: `${mc_stats['min_price']:,.2f}`")
    st.write(f"Max Simulated Price: `${mc_stats['max_price']:,.2f}`")
