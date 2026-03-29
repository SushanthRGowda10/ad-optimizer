import pandas as pd
import numpy as np


def summary_stats(df: pd.DataFrame) -> dict:
    """High-level KPI summary for the dashboard."""
    return {
        'total_campaigns': int(df['campaign_name'].nunique()),
        'total_impressions': int(df['impressions'].sum()),
        'total_clicks': int(df['clicks'].sum()),
        'total_conversions': int(df['conversions'].sum()),
        'total_cost': round(float(df['cost'].sum()), 2),
        'avg_ctr': round(float(df['ctr'].mean() * 100), 4),
        'avg_cpc': round(float(df['cpc'].mean()), 4),
        'avg_conversion_rate': round(float(df['conversion_rate'].mean() * 100), 4),
    }


def top_campaigns(df: pd.DataFrame, n: int = 5) -> list:
    """Return top N campaigns by total clicks."""
    grouped = (
        df.groupby('campaign_name')
          .agg(impressions=('impressions', 'sum'),
               clicks=('clicks', 'sum'),
               conversions=('conversions', 'sum'),
               cost=('cost', 'sum'))
          .reset_index()
    )
    grouped['ctr'] = (grouped['clicks'] / grouped['impressions'].replace(0, np.nan)).fillna(0) * 100
    return grouped.nlargest(n, 'clicks').to_dict('records')


def platform_performance(df: pd.DataFrame) -> list:
    grouped = (
        df.groupby('platform')
          .agg(impressions=('impressions', 'sum'),
               clicks=('clicks', 'sum'),
               conversions=('conversions', 'sum'),
               cost=('cost', 'sum'))
          .reset_index()
    )
    grouped['ctr'] = (grouped['clicks'] / grouped['impressions'].replace(0, np.nan)).fillna(0) * 100
    return grouped.sort_values('clicks', ascending=False).to_dict('records')


def age_group_performance(df: pd.DataFrame) -> list:
    grouped = (
        df.groupby('age_group')
          .agg(clicks=('clicks', 'sum'),
               conversions=('conversions', 'sum'),
               cost=('cost', 'sum'))
          .reset_index()
    )
    return grouped.sort_values('conversions', ascending=False).to_dict('records')


def device_performance(df: pd.DataFrame) -> list:
    grouped = (
        df.groupby('device')
          .agg(impressions=('impressions', 'sum'),
               clicks=('clicks', 'sum'),
               conversions=('conversions', 'sum'),
               cost=('cost', 'sum'))
          .reset_index()
    )
    grouped['ctr'] = (grouped['clicks'] / grouped['impressions'].replace(0, np.nan)).fillna(0) * 100
    return grouped.sort_values('clicks', ascending=False).to_dict('records')


def location_performance(df: pd.DataFrame) -> list:
    grouped = (
        df.groupby('location')
          .agg(clicks=('clicks', 'sum'),
               conversions=('conversions', 'sum'),
               cost=('cost', 'sum'))
          .reset_index()
    )
    return grouped.nlargest(10, 'conversions').to_dict('records')


def ctr_trend(df: pd.DataFrame) -> dict:
    """Weekly CTR trend for line chart."""
    df = df.copy()
    df['week_start'] = pd.to_datetime(df['date']).dt.to_period('W').apply(lambda r: str(r.start_time.date()))
    trend = (
        df.groupby('week_start')
          .apply(lambda g: (g['clicks'].sum() / g['impressions'].sum() * 100) if g['impressions'].sum() > 0 else 0)
          .reset_index()
          .rename(columns={0: 'ctr'})
          .sort_values('week_start')
    )
    return {'labels': trend['week_start'].tolist(), 'values': trend['ctr'].round(4).tolist()}


def conversion_trend(df: pd.DataFrame) -> dict:
    df = df.copy()
    df['week_start'] = pd.to_datetime(df['date']).dt.to_period('W').apply(lambda r: str(r.start_time.date()))
    trend = (
        df.groupby('week_start')['conversions']
          .sum()
          .reset_index()
          .sort_values('week_start')
    )
    return {'labels': trend['week_start'].tolist(), 'values': trend['conversions'].tolist()}
