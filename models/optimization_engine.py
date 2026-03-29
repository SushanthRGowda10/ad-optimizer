import pandas as pd
import numpy as np


def generate_recommendations(df: pd.DataFrame) -> dict:
    """
    Rule-based + ML-assisted optimizer.
    Returns structured recommendations and insights.
    """
    recs = {}

    # --- Best Platform ---
    plat = df.groupby('platform').agg(
        clicks=('clicks', 'sum'),
        conversions=('conversions', 'sum'),
        impressions=('impressions', 'sum'),
        cost=('cost', 'sum')
    ).reset_index()
    plat['ctr'] = plat['clicks'] / plat['impressions'].replace(0, np.nan)
    plat['conv_rate'] = plat['conversions'] / plat['clicks'].replace(0, np.nan)
    best_plat = plat.sort_values('conv_rate', ascending=False).iloc[0]
    recs['best_platform'] = str(best_plat['platform'])

    # --- Best Age Group ---
    age = df.groupby('age_group').agg(
        conversions=('conversions', 'sum'),
        clicks=('clicks', 'sum')
    ).reset_index()
    age['conv_rate'] = age['conversions'] / age['clicks'].replace(0, np.nan)
    best_age = age.sort_values('conv_rate', ascending=False).iloc[0]
    recs['best_age_group'] = str(best_age['age_group'])

    # --- Best Device ---
    dev = df.groupby('device').agg(
        ctr=('ctr', 'mean'),
        conversions=('conversions', 'sum')
    ).reset_index()
    best_dev = dev.sort_values('conversions', ascending=False).iloc[0]
    recs['best_device'] = str(best_dev['device'])

    # --- Best Campaign ---
    camp = df.groupby('campaign_name').agg(
        clicks=('clicks', 'sum'),
        conversions=('conversions', 'sum'),
        cost=('cost', 'sum'),
        impressions=('impressions', 'sum')
    ).reset_index()
    camp['perf'] = camp['conversions'] / (camp['cost'].replace(0, np.nan))
    best_camp = camp.sort_values('perf', ascending=False).iloc[0]
    recs['best_campaign'] = str(best_camp['campaign_name'])

    # --- Increase Budget: High CTR, low spend ---
    camp['ctr'] = camp['clicks'] / camp['impressions'].replace(0, np.nan)
    high_ctr = camp[camp['ctr'] > camp['ctr'].median()].sort_values('cost').head(3)
    recs['increase_budget'] = high_ctr['campaign_name'].tolist()

    # --- Pause: Low conversion rate and high cost ---
    camp['conv_rate'] = camp['conversions'] / camp['clicks'].replace(0, np.nan)
    poor = camp[
        (camp['conv_rate'] < camp['conv_rate'].quantile(0.25)) &
        (camp['cost'] > camp['cost'].median())
    ]
    recs['pause_campaigns'] = poor['campaign_name'].tolist()

    # --- Budget Shift Suggestion ---
    recs['budget_insight'] = (
        f"Shift 20% of budget from underperforming campaigns to "
        f"{recs['best_platform']} targeting {recs['best_age_group']} on {recs['best_device']}."
    )

    # --- Platform insights list ---
    plat_sorted = plat.sort_values('conv_rate', ascending=False)
    recs['platform_insights'] = [
        {
            'platform': row['platform'],
            'ctr': round(float(row['ctr']) * 100, 3) if not pd.isna(row['ctr']) else 0,
            'conv_rate': round(float(row['conv_rate']) * 100, 3) if not pd.isna(row['conv_rate']) else 0,
            'total_cost': round(float(row['cost']), 2),
        }
        for _, row in plat_sorted.iterrows()
    ]

    # --- Optimization Score (0-100) ---
    avg_conv = float(df['conversion_rate'].mean()) if 'conversion_rate' in df.columns else 0
    recs['optimization_score'] = min(100, round(avg_conv * 1000, 1))

    return recs
