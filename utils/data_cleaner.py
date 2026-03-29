import pandas as pd
import numpy as np

REQUIRED_COLUMNS = ['ad_id', 'campaign_name', 'platform', 'impressions',
                    'clicks', 'conversions', 'cost', 'age_group',
                    'location', 'device', 'date']

NUMERIC_COLUMNS = ['impressions', 'clicks', 'conversions', 'cost']


def validate_columns(df):
    """Check that all required columns are present."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def clean_dataset(df: pd.DataFrame) -> dict:
    """
    Full cleaning pipeline. Returns cleaned DataFrame + report.
    """
    report = {
        'original_rows': len(df),
        'duplicates_removed': 0,
        'nulls_filled': 0,
        'outliers_capped': 0,
    }

    # 1. Validate columns
    validate_columns(df)

    # 2. Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    report['duplicates_removed'] = before - len(df)

    # 3. Coerce numeric types
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Fill missing numeric values with median
    null_before = df[NUMERIC_COLUMNS].isnull().sum().sum()
    for col in NUMERIC_COLUMNS:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0)
    report['nulls_filled'] = int(null_before)

    # 5. Fill missing categorical values
    for col in ['platform', 'age_group', 'location', 'device', 'campaign_name']:
        df[col] = df[col].fillna('Unknown')

    # 6. Ensure non-negative values
    for col in NUMERIC_COLUMNS:
        df[col] = df[col].clip(lower=0)

    # 7. Parse dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = df['date'].fillna(pd.Timestamp.now())

    # 8. Cap outliers using IQR (1.5x)
    outlier_count = 0
    for col in NUMERIC_COLUMNS:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_count += outliers
        df[col] = df[col].clip(lower=max(0, lower), upper=upper)
    report['outliers_capped'] = int(outlier_count)

    report['final_rows'] = len(df)
    return {'data': df, 'report': report}
