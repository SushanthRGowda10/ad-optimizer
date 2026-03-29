import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute key advertising metrics and add as new columns.
    CTR, CPC, Conversion Rate, ROAS, CPM.
    """
    df = df.copy()

    # Click-Through Rate: clicks per impression
    df['ctr'] = df.apply(
        lambda r: r['clicks'] / r['impressions'] if r['impressions'] > 0 else 0, axis=1
    )

    # Cost Per Click
    df['cpc'] = df.apply(
        lambda r: r['cost'] / r['clicks'] if r['clicks'] > 0 else 0, axis=1
    )

    # Conversion Rate
    df['conversion_rate'] = df.apply(
        lambda r: r['conversions'] / r['clicks'] if r['clicks'] > 0 else 0, axis=1
    )

    # Cost Per Mille (per 1000 impressions)
    df['cpm'] = df.apply(
        lambda r: (r['cost'] / r['impressions']) * 1000 if r['impressions'] > 0 else 0, axis=1
    )

    # Cost Per Acquisition
    df['cpa'] = df.apply(
        lambda r: r['cost'] / r['conversions'] if r['conversions'] > 0 else 0, axis=1
    )

    # Composite performance score (0–100)
    # Normalize each metric and weight
    def safe_norm(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - mn) / (mx - mn)

    df['perf_score'] = (
        safe_norm(df['ctr']) * 35 +
        safe_norm(df['conversion_rate']) * 35 +
        safe_norm(1 / (df['cpc'] + 0.01)) * 30
    ).round(2)

    # Time features
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['week'] = pd.to_datetime(df['date']).dt.isocalendar().week.astype(int)

    return df
