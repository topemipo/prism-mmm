import os
import requests
import streamlit as st

from allocation import marginal_roi
from app_config import load_model_config

CONFIG = load_model_config()
MEDIA_COLS = CONFIG['media_cols']
CHANNEL_LABELS = CONFIG['channel_labels']
HILL_PARAMS = CONFIG['hill_params']
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def build_recommendation_prompt(scenario_result, roi_table, coefs, channel_state):
    """Returns (system_message, user_message) for the chat API."""
    import numpy as np
    summary   = scenario_result['summary']
    comp_df   = scenario_result['comp_df']
    opt_spend = np.array(scenario_result['raw']['spend'])

    mroi_lines = []
    for i, col in enumerate(MEDIA_COLS):
        p = HILL_PARAMS[col]
        m = marginal_roi(opt_spend[i], coefs[col], p['K'], p['S'],
                         channel_state[col]['adstock_max'], p['decay'])
        mroi_lines.append(f"  {CHANNEL_LABELS[col]}: {m:.2f}x")

    system_msg = "You are a media planning expert advising a marketing director."
    user_msg = (
        f"A Marketing Mix Model has produced these budget optimisation results.\n\n"
        f"## Historical Channel ROI\n{roi_table.to_string(index=False)}\n\n"
        f"## Optimised vs Current Allocation\n{comp_df.to_string(index=False)}\n\n"
        f"## Marginal ROI at Optimised Spend\n{chr(10).join(mroi_lines)}\n\n"
        f"## Summary\n"
        f"Target budget: £{summary['target_budget']:,.0f}/week\n"
        f"Predicted response: £{summary['new_total_response']:,.0f}/week\n"
        f"Uplift vs current: £{summary['uplift']:,.0f} ({summary['uplift_pct']:+.1f}%)\n\n"
        f"Write 4–5 bullet-point recommendations for the marketing director. "
        f"Be specific with numbers. Include: which channels to prioritise and why, "
        f"which to cut and why, any diminishing-returns risks. "
        f"Start with a one-sentence headline."
    )
    return system_msg, user_msg


def get_rule_based_recommendation(scenario_result, coefs, channel_state):
    import numpy as np
    summary   = scenario_result['summary']
    comp_df   = scenario_result['comp_df']
    opt_spend = np.array(scenario_result['raw']['spend'])

    mroi_data = []
    for i, col in enumerate(MEDIA_COLS):
        p = HILL_PARAMS[col]
        m = marginal_roi(opt_spend[i], coefs[col], p['K'], p['S'],
                         channel_state[col]['adstock_max'], p['decay'])
        mroi_data.append({'label': CHANNEL_LABELS[col], 'mroi': m})
    mroi_data.sort(key=lambda x: x['mroi'], reverse=True)

    increases = comp_df[comp_df['Change %'] > 5.0].sort_values('Change %', ascending=False)
    decreases = comp_df[comp_df['Change %'] < -5.0].sort_values('Change %')

    lines = [
        f"**Rebalancing to £{summary['target_budget']:,.0f}/wk delivers a predicted "
        f"uplift of £{summary['uplift']:,.0f} ({summary['uplift_pct']:+.1f}%).**\n"
    ]
    for _, row in increases.iterrows():
        lines.append(
            f"- **Increase {row['Channel']}** by {row['Change %']:+.1f}% "
            f"(£{row['Current Spend']:,.0f} → £{row['Recommended Spend']:,.0f}/wk) — "
            f"response rises from £{row['Current Response']:,.0f} to £{row['New Response']:,.0f}."
        )
    for _, row in decreases.iterrows():
        lines.append(
            f"- **Reduce {row['Channel']}** by {abs(row['Change %']):.1f}% "
            f"(£{row['Current Spend']:,.0f} → £{row['Recommended Spend']:,.0f}/wk) — "
            f"capital freed for higher-ROI channels."
        )
    lines.append(
        f"\n- **Highest marginal ROI at optimised spend**: {mroi_data[0]['label']} "
        f"({mroi_data[0]['mroi']:.2f}x). Consider widening its upper bound."
    )
    lines.append(
        f"- **Lowest marginal ROI**: {mroi_data[-1]['label']} ({mroi_data[-1]['mroi']:.2f}x). "
        f"Diminishing returns are strongest here."
    )
    return "\n".join(lines)


def get_ai_recommendation(scenario_result, roi_table, coefs, channel_state):
    groq_key = None
    try:
        groq_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass
    if not groq_key:
        groq_key = os.environ.get("GROQ_API_KEY")

    if not groq_key:
        return get_rule_based_recommendation(scenario_result, coefs, channel_state)

    system_msg, user_msg = build_recommendation_prompt(scenario_result, roi_table, coefs, channel_state)
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                "max_tokens": 500,
                "temperature": 0.3,
            },
            timeout=45,
        )
        if resp.status_code == 503:
            return (get_rule_based_recommendation(scenario_result, coefs, channel_state)
                    + "\n\n_Model is loading — try again in a moment._")
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return text if text else get_rule_based_recommendation(scenario_result, coefs, channel_state)
    except Exception as e:
        return (get_rule_based_recommendation(scenario_result, coefs, channel_state)
                + f"\n\n_AI unavailable: {e}_")
