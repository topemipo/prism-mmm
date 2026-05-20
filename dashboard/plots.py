import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from allocation import response as ch_response, marginal_roi, total_response as alloc_total_response
from app_config import load_model_config

CONFIG = load_model_config()
MEDIA_COLS = CONFIG['media_cols']
CHANNEL_LABELS = CONFIG['channel_labels']
HILL_PARAMS = CONFIG['hill_params']
SCENARIO_COLOURS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']


def _hex_to_rgba(hex_color, alpha=1.0):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def _lighten(hex_color, factor=0.62):
    """Mix hex_color with white. factor=0 → original, factor=1 → white."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f'rgb({r},{g},{b})'


def plot_overview_bars(scenario_results, current_spend, coefs, channel_state):
    cur_resp  = alloc_total_response(current_spend, MEDIA_COLS, coefs, HILL_PARAMS, channel_state)
    labels    = ['Current'] + [s['summary']['scenario'] for s in scenario_results]
    spends    = [float(current_spend.sum())] + [s['summary']['target_budget'] for s in scenario_results]
    responses = [cur_resp] + [s['summary']['new_total_response'] for s in scenario_results]
    colours   = ['#aaaaaa'] + SCENARIO_COLOURS[:len(scenario_results)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Total Spend (£)', x=labels, y=spends,
        marker_color=[_hex_to_rgba(c, 0.45) for c in colours],
        text=[f'£{v:,.0f}' for v in spends],
        textposition='outside',
        offsetgroup=0,
    ))
    fig.add_trace(go.Bar(
        name='Predicted Response (£)', x=labels, y=responses,
        marker_color=colours,
        text=[f'£{v:,.0f}' for v in responses],
        textposition='outside',
        offsetgroup=1,
    ))
    fig.update_layout(
        barmode='group',
        title='Total Spend vs Predicted Response by Scenario',
        yaxis_title='£ / week',
        height=440,
        margin=dict(t=80, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


def _make_heatmap_matrix(spend_array, coefs, channel_state):
    spend_array = np.array(spend_array)
    total_spend = spend_array.sum()
    resp_vals, mroi_vals = [], []

    for i, col in enumerate(MEDIA_COLS):
        p = HILL_PARAMS[col]
        s = channel_state[col]
        r = ch_response(spend_array[i], p['decay'], coefs[col], p['K'], p['S'], s['adstock_max'])
        m = marginal_roi(spend_array[i], coefs[col], p['K'], p['S'], s['adstock_max'], p['decay'])
        resp_vals.append(r)
        mroi_vals.append(m)

    total_resp = sum(resp_vals)
    z     = np.zeros((4, len(MEDIA_COLS)))
    annot = [[''] * len(MEDIA_COLS) for _ in range(4)]

    for i in range(len(MEDIA_COLS)):
        sp_pct   = spend_array[i] / total_spend * 100 if total_spend > 0 else 0
        resp_pct = resp_vals[i] / total_resp * 100 if total_resp > 0 else 0
        roas     = resp_vals[i] / spend_array[i] if spend_array[i] > 0 else 0

        z[0, i] = sp_pct;                   annot[0][i] = f'{sp_pct:.1f}%'
        z[1, i] = resp_pct;                 annot[1][i] = f'{resp_pct:.1f}%'
        z[2, i] = min(roas, 100);           annot[2][i] = f'{roas:.2f}x'
        z[3, i] = min(mroi_vals[i], 100);   annot[3][i] = f'{mroi_vals[i]:.2f}x'

    return z, annot


def plot_channel_heatmaps(scenario_results, current_spend, coefs, channel_state):
    n_panels      = 1 + len(scenario_results)
    titles        = ['Current'] + [f"Scenario {i+1}" for i in range(len(scenario_results))]
    all_spends    = [current_spend] + [np.array(s['raw']['spend']) for s in scenario_results]
    ch_names      = [CHANNEL_LABELS[c] for c in MEDIA_COLS]
    metric_names  = ['Spend %', 'Response %', 'ROAS', 'mROAS']
    panel_colours = ['#aaaaaa'] + SCENARIO_COLOURS[:len(scenario_results)]

    fig = make_subplots(
        rows=1, cols=n_panels,
        subplot_titles=titles,
        horizontal_spacing=0.03,
    )

    for panel_idx, (spend_arr, colour) in enumerate(zip(all_spends, panel_colours)):
        _, annot = _make_heatmap_matrix(spend_arr, coefs, channel_state)
        light    = _lighten(colour, 0.62)
        flat_z   = np.ones((4, len(MEDIA_COLS)))

        fig.add_trace(go.Heatmap(
            z=flat_z,
            x=ch_names,
            y=metric_names,
            colorscale=[[0, light], [1, light]],
            showscale=False,
            text=annot,
            texttemplate='%{text}',
            hoverinfo='text',
            xgap=3,
            ygap=3,
        ), row=1, col=panel_idx + 1)

        if panel_idx > 0:
            fig.update_yaxes(showticklabels=False, row=1, col=panel_idx + 1)

    fig.update_layout(
        title='Budget Allocation per Channel',
        height=340, margin=dict(t=80, b=20),
    )
    return fig


def plot_response_curves(scenario_results, current_spend, coefs, channel_state):
    ch_names = [CHANNEL_LABELS[c] for c in MEDIA_COLS]
    fig = make_subplots(rows=1, cols=5, subplot_titles=ch_names, horizontal_spacing=0.07)

    all_labels  = ['Current'] + [f"Scenario {i+1}" for i in range(len(scenario_results))]
    all_spends  = [current_spend] + [np.array(s['raw']['spend']) for s in scenario_results]
    dot_colours = ['#444444'] + SCENARIO_COLOURS[:len(scenario_results)]
    legend_seen = set()

    n_dots = len(all_labels)
    _half  = (n_dots - 1) / 2
    NUDGE_FRACTIONS = [(i - _half) * 0.012 for i in range(n_dots)]

    for ch_idx, col in enumerate(MEDIA_COLS):
        p    = HILL_PARAMS[col]
        amax = channel_state[col]['adstock_max']
        coef = coefs[col]

        max_dot = max(float(sp[ch_idx]) for sp in all_spends)
        x_max   = max_dot * 1.6
        x_curve = np.linspace(0, x_max, 100)
        y_curve = [ch_response(x, p['decay'], coef, p['K'], p['S'], amax) for x in x_curve]

        fig.add_trace(go.Scatter(
            x=x_curve, y=y_curve, mode='lines',
            line=dict(color='#aaaaaa', width=2),
            showlegend=False, hoverinfo='skip',
        ), row=1, col=ch_idx + 1)

        for sc_idx, (label, spend_arr, colour, nudge) in enumerate(
            zip(all_labels, all_spends, dot_colours, NUDGE_FRACTIONS)
        ):
            true_spend = float(spend_arr[ch_idx])
            nudged_x   = true_spend + nudge * x_max
            true_resp  = ch_response(true_spend, p['decay'], coef, p['K'], p['S'], amax)

            show = label not in legend_seen
            if show:
                legend_seen.add(label)

            if sc_idx == 0:
                marker = dict(
                    size=13, symbol='circle-open',
                    color=colour, line=dict(width=3, color=colour),
                )
            else:
                marker = dict(size=13, color=colour, line=dict(width=2.5, color='white'))

            fig.add_trace(go.Scatter(
                x=[nudged_x], y=[true_resp], mode='markers',
                marker=marker,
                name=label, showlegend=show, legendgroup=label,
                customdata=[[true_spend]],
                hovertemplate=(
                    f'{label}<br>Spend: £%{{customdata[0]:,.0f}}'
                    f'<br>Response: £%{{y:,.0f}}<extra></extra>'
                ),
            ), row=1, col=ch_idx + 1)

    fig.update_layout(
        title='Response Curves — Where Each Scenario Sits',
        height=430, margin=dict(t=80, b=110),
        legend=dict(orientation='h', yanchor='bottom', y=-0.30, xanchor='center', x=0.5),
    )
    for i in range(1, 6):
        fig.update_xaxes(tickformat='.2s', row=1, col=i)
        fig.update_yaxes(tickformat='.2s', row=1, col=i)

    return fig
