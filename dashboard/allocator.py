import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from model import build_design_matrix, fit_ols
from transformations import geometric_adstock
from allocation import run_optimisation, build_comparison
from plots import plot_overview_bars, plot_channel_heatmaps, plot_response_curves
from ai import get_ai_recommendation
from app_config import load_model_config

# ── Constants ─────────────────────────────────────────────────────────────────
CONFIG = load_model_config()
MEDIA_COLS = CONFIG['media_cols']
CHANNEL_LABELS = CONFIG['channel_labels']
HILL_PARAMS = CONFIG['hill_params']
SAT_COLS = CONFIG['sat_cols']
CONTROL_COLS = CONFIG['control_cols']
DEFAULT_WEEKLY_BUDGET = CONFIG['default_weekly_budget']
SCENARIO_COLOURS      = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']

_DEFAULT_BOUNDS_DF = pd.DataFrame({
    'Channel':        [CHANNEL_LABELS[c] for c in MEDIA_COLS],
    'Max cut %':      [30, 30, 30, 30, 30],
    'Max increase %': [30, 30, 30, 30, 30],
})
_BUDGET_DEFAULTS = [int(DEFAULT_WEEKLY_BUDGET * m) for m in [1.0, 1.2, 0.8, 1.4, 0.6, 1.6]]

st.set_page_config(page_title="Media Budget Optimiser", page_icon="🧭", layout="wide")


# ── Cached data loader ────────────────────────────────────────────────────────
@st.cache_data
def load_model_and_data():
    df_trans  = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'dt_transformed.csv'))
    df_raw    = pd.read_csv(os.path.join(BASE_DIR, 'data', 'raw', 'dt_simulated_weekly.csv'))
    df_raw    = df_raw.sort_values('DATE').reset_index(drop=True)
    roi_table = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'roi_table.csv'))

    available_ctrl = [c for c in CONTROL_COLS if c in df_trans.columns]
    X       = build_design_matrix(df_trans, SAT_COLS, available_ctrl)
    y       = df_trans['revenue'].reset_index(drop=True)
    results = fit_ols(X, y)

    coefs = {col: float(results.params[sat]) for col, sat in zip(MEDIA_COLS, SAT_COLS)}

    channel_state = {}
    for col in MEDIA_COLS:
        raw = df_raw[col].values
        ads = geometric_adstock(raw, HILL_PARAMS[col]['decay'])
        channel_state[col] = {
            'mean_weekly_spend': float(raw.mean()),
            'adstock_max':       float(ads.max()),
        }

    current_spend = np.array([channel_state[col]['mean_weekly_spend'] for col in MEDIA_COLS])
    return roi_table, coefs, channel_state, current_spend, float(results.rsquared)


# ── Optimisation helpers ──────────────────────────────────────────────────────
def build_per_channel_bounds(current_spend, cut_pcts, increase_pcts):
    return [
        (max(0.0, current_spend[i] * (1 - cut_pcts[col] / 100.0)),
         current_spend[i] * (1 + increase_pcts[col] / 100.0))
        for i, col in enumerate(MEDIA_COLS)
    ]


def check_bounds_feasibility(bounds, target_budget):
    lo_sum = sum(lo for lo, _ in bounds)
    hi_sum = sum(hi for _, hi in bounds)
    if lo_sum > target_budget:
        return False, f"Lower bounds require at least £{lo_sum:,.0f}/wk — raise budget or loosen lower bounds."
    if hi_sum < target_budget:
        return False, f"Upper bounds cap at £{hi_sum:,.0f}/wk — lower budget or raise upper bounds."
    return True, ""


def run_all_scenarios(scenario_configs, current_spend, coefs, channel_state):
    results = []
    for cfg in scenario_configs:
        if not cfg['feasible']:
            continue
        raw = run_optimisation(
            target_budget=cfg['budget'],
            scenario_label=cfg['label'],
            current_spend=current_spend,
            bounds=cfg['bounds'],
            media_cols=MEDIA_COLS,
            coefs=coefs,
            hill_params=HILL_PARAMS,
            channel_state=channel_state,
        )
        comp_df, summary = build_comparison(
            scenario=raw,
            current_spend=current_spend,
            media_cols=MEDIA_COLS,
            coefs=coefs,
            hill_params=HILL_PARAMS,
            channel_state=channel_state,
            channel_labels=CHANNEL_LABELS,
        )
        results.append({
            'raw':        raw,
            'comp_df':    comp_df,
            'summary':    summary,
            'bounds':     cfg['bounds'],
            'lower_pcts': cfg['lower_pcts'],
            'upper_pcts': cfg['upper_pcts'],
        })
    return results


# ── Results renderer ──────────────────────────────────────────────────────────
def render_results(scenario_results, current_spend, coefs, channel_state, roi_table):
    st.divider()
    st.header("Optimisation Results")

    st.subheader("Spend & Response by Scenario")
    st.plotly_chart(
        plot_overview_bars(scenario_results, current_spend, coefs, channel_state),
        use_container_width=True,
    )

    st.subheader("Channel Efficiency Heatmap")
    st.caption("Spend % and Response % are shares of weekly total. ROAS and mROAS are £ per £.")
    st.plotly_chart(
        plot_channel_heatmaps(scenario_results, current_spend, coefs, channel_state),
        use_container_width=True,
    )

    st.subheader("Response Curves")
    st.caption("Each dot shows where a scenario's optimised spend sits on the channel's response curve.")
    st.plotly_chart(
        plot_response_curves(scenario_results, current_spend, coefs, channel_state),
        use_container_width=True,
    )

    st.subheader("Scenario Detail")
    for idx, sc in enumerate(scenario_results):
        summary = sc['summary']
        header  = (
            f"Scenario {idx+1} — {summary['scenario']}  ·  "
            f"Uplift: **£{summary['uplift']:,.0f}** ({summary['uplift_pct']:+.1f}%)"
        )
        with st.expander(header, expanded=(idx == 0)):
            m1, m2, m3 = st.columns(3)
            m1.metric("Budget",             f"£{summary['target_budget']:,.0f}/wk")
            m2.metric("Predicted Response", f"£{summary['new_total_response']:,.0f}/wk")
            m3.metric("Uplift vs Current",  f"{summary['uplift_pct']:+.1f}%")

            st.markdown("**Optimised allocation vs current**")
            disp = sc['comp_df'].copy()
            for col_name in ['Current Spend', 'Recommended Spend', 'Current Response', 'New Response']:
                disp[col_name] = disp[col_name].apply(lambda x: f'£{x:,.0f}')
            disp['Change %'] = disp['Change %'].apply(lambda x: f'{x:+.1f}%')
            st.dataframe(disp, hide_index=True, use_container_width=True)

            st.markdown("---")
            rec_key = f"ai_rec_{idx}"
            if st.button("Generate Recommendation", key=f"ai_btn_{idx}", type="secondary"):
                with st.spinner("Generating..."):
                    rec = get_ai_recommendation(sc, roi_table, coefs, channel_state)
                st.session_state[rec_key] = rec

            if rec_key in st.session_state:
                st.markdown(st.session_state[rec_key])


# ── Load data ─────────────────────────────────────────────────────────────────
roi_table, coefs, channel_state, current_spend, r_squared = load_model_and_data()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Context")

    _total_spend_hist = roi_table['Total Spend'].sum()
    _total_rev_hist   = roi_table['Revenue Contribution'].sum()
    _blended_roi      = _total_rev_hist / _total_spend_hist

    st.markdown("**Historical media performance (208 weeks)**")
    m1, m2 = st.columns(2)
    m1.metric("Total Media Spend",  f"£{_total_spend_hist/1e6:.2f}M")
    m2.metric("Revenue from Media", f"£{_total_rev_hist/1e6:.2f}M")
    m3, m4 = st.columns(2)
    m3.metric("Blended Media ROI", f"{_blended_roi:.2f}x",
              help="Total media-attributed revenue ÷ total media spend across all channels")
    m4.metric("Model R²", f"{r_squared:.4f}",
              help="How well the MMM explains observed revenue variance. Closer to 1.0 = better fit.")

    st.divider()

    roi_sorted = roi_table.sort_values('ROI', ascending=True)
    fig_roi = go.Figure(go.Bar(
        x=roi_sorted['ROI'], y=roi_sorted['Channel'],
        orientation='h', marker_color='#4a90d9',
        text=[f"{r:.2f}x" for r in roi_sorted['ROI']],
        textposition='outside',
    ))
    fig_roi.update_layout(
        title="Historical ROI by Channel", height=220,
        margin=dict(l=5, r=50, t=40, b=10),
        xaxis_title="£ per £ spent", showlegend=False,
    )
    st.plotly_chart(fig_roi, use_container_width=True)

    total_spend_hist = roi_table['Total Spend'].sum()
    total_rev_hist   = roi_table['Revenue Contribution'].sum()
    fig_share = go.Figure()
    fig_share.add_trace(go.Bar(
        name='Budget %', x=roi_table['Channel'],
        y=(roi_table['Total Spend'] / total_spend_hist * 100).round(1),
        marker_color='#aec7e8',
    ))
    fig_share.add_trace(go.Bar(
        name='Revenue %', x=roi_table['Channel'],
        y=(roi_table['Revenue Contribution'] / total_rev_hist * 100).round(1),
        marker_color='#1f77b4',
    ))
    fig_share.update_layout(
        barmode='group', title="Budget vs Revenue Share",
        height=250, margin=dict(l=5, r=5, t=40, b=30),
        yaxis_title="%", legend=dict(orientation='h', y=-0.18),
    )
    st.plotly_chart(fig_share, use_container_width=True)

    st.divider()
    st.markdown("**Weekly averages by channel (208 weeks)**")
    _weekly = pd.DataFrame({
        'Channel':        roi_table['Channel'].values,
        'Avg Spend':      (roi_table['Total Spend'] / 208).apply(lambda x: f"£{x:,.0f}"),
        'Avg Revenue':    (roi_table['Revenue Contribution'] / 208).apply(lambda x: f"£{x:,.0f}"),
    })
    st.dataframe(_weekly, hide_index=True, use_container_width=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align: center;'>Media Budget Optimiser</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>"
    "<a href='https://github.com/topemipo/prism-mmm' target='_blank'>GitHub Repository</a>"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "Each scenario has its own **budget** and its own **per-channel bounds**. "
    "The optimiser finds the revenue-maximising spend split within those constraints. "
    "All scenarios are compared against the current baseline."
)

_col_label, _col_input = st.columns([3, 1])
_col_label.markdown("**Number of scenarios to configure**")
n_scenarios = _col_input.number_input(
    "scenarios", min_value=1, max_value=6, value=1, step=1, label_visibility="collapsed",
)

st.subheader("Scenario Configuration")

scenario_configs = []

for i in range(int(n_scenarios)):
    with st.expander(f"Scenario {i + 1}", expanded=(i == 0)):

        budget = st.number_input(
            "Weekly Budget (£)",
            min_value=5_000, max_value=500_000,
            value=_BUDGET_DEFAULTS[i], step=500,
            key=f"budget_sc{i}",
            help="Total media budget for this scenario per week",
        )

        st.markdown(
            "**Channel bounds** — how much freedom the optimiser has per channel. "
            "e.g. Max cut = 30 means spend can drop by at most 30% of current. "
            "Max increase = 30 means spend can rise by at most 30% of current."
        )
        edited_bounds = st.data_editor(
            _DEFAULT_BOUNDS_DF.copy(),
            key=f"bounds_df_sc{i}",
            disabled=['Channel'],
            hide_index=True,
            use_container_width=True,
            column_config={
                'Channel': st.column_config.TextColumn("Channel", width="small"),
                'Max cut %': st.column_config.NumberColumn(
                    "Max cut %", min_value=0, max_value=100, step=5,
                    help="Maximum % you allow this channel's spend to be reduced. "
                         "e.g. 30 → floor = 70% of current spend.",
                ),
                'Max increase %': st.column_config.NumberColumn(
                    "Max increase %", min_value=0, max_value=200, step=5,
                    help="Maximum % you allow this channel's spend to grow. "
                         "e.g. 30 → ceiling = 130% of current spend.",
                ),
            },
        )

        cut_pcts      = {MEDIA_COLS[j]: int(edited_bounds.iloc[j]['Max cut %'])      for j in range(5)}
        increase_pcts = {MEDIA_COLS[j]: int(edited_bounds.iloc[j]['Max increase %']) for j in range(5)}
        bounds        = build_per_channel_bounds(current_spend, cut_pcts, increase_pcts)

        preview = pd.DataFrame({
            'Channel':  [CHANNEL_LABELS[c] for c in MEDIA_COLS],
            'Current':  [f"£{current_spend[j]:,.0f}" for j in range(5)],
            'Min £/wk': [f"£{bounds[j][0]:,.0f}" for j in range(5)],
            'Max £/wk': [f"£{bounds[j][1]:,.0f}" for j in range(5)],
        })
        st.dataframe(preview, hide_index=True, use_container_width=True)

        feasible, feas_msg = check_bounds_feasibility(bounds, budget)
        if not feasible:
            st.warning(f"⚠ {feas_msg}")

        scenario_configs.append({
            'label':      f"Scenario {i + 1} — £{budget:,.0f}/wk",
            'budget':     float(budget),
            'bounds':     bounds,
            'lower_pcts': cut_pcts,
            'upper_pcts': increase_pcts,
            'feasible':   feasible,
        })

feasible_count   = sum(1 for c in scenario_configs if c['feasible'])
infeasible_count = len(scenario_configs) - feasible_count
if infeasible_count > 0:
    st.info(
        f"{feasible_count} of {len(scenario_configs)} scenario(s) are feasible and will be optimised. "
        f"{infeasible_count} will be skipped."
    )

st.divider()
run_clicked = st.button(
    "Run Optimisation", type="primary",
    use_container_width=True,
    disabled=(feasible_count == 0),
)

if run_clicked:
    with st.spinner(f"Optimising {feasible_count} scenario(s)..."):
        results = run_all_scenarios(scenario_configs, current_spend, coefs, channel_state)
    st.session_state["scenario_results"] = results

if "scenario_results" in st.session_state:
    render_results(
        st.session_state["scenario_results"],
        current_spend, coefs, channel_state, roi_table,
    )
