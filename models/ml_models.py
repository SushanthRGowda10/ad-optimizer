import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

CATEGORICAL_COLS = ['platform', 'age_group', 'device', 'campaign_name', 'location']
FEATURE_COLS = ['impressions', 'cost', 'platform_enc', 'age_group_enc',
                'device_enc', 'month', 'day_of_week']




def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns."""
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def train_models(df: pd.DataFrame) -> dict:
    """
    Train Linear Regression, Decision Tree, and Random Forest
    on clicks and conversions. Return metrics and model objects.
    """
    df, encoders = _encode(df)

    results = {}

    for target in ['clicks', 'conversions']:
        X = df[FEATURE_COLS].fillna(0)
        y = df[target].fillna(0)

        if len(X) < 10:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        }

        target_results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = round(r2_score(y_test, y_pred), 4)
            mae = round(mean_absolute_error(y_test, y_pred), 2)
            target_results[name] = {
                'r2': max(0, r2),
                'mae': mae,
                'model': model
            }

        results[target] = target_results

    return {'results': results, 'encoders': encoders}


def predict_performance(platform: str, age_group: str, device: str,
                        budget: float, impressions: float,
                        trained: dict) -> dict:
    """
    Predict clicks and conversions given campaign parameters.
    Uses Random Forest from the trained model bundle.
    """
    results = trained.get('results', {})
    encoders = trained.get('encoders', {})


    def encode_val(col, val):
        le = encoders.get(col)
        if le is None:
            return 0
        try:
            return int(le.transform([val])[0])
        except ValueError:
            return 0

    features = np.array([[
        impressions,
        budget,
        encode_val('platform', platform),
        encode_val('age_group', age_group),
        encode_val('device', device),
        6,   # month (June - mid-year default)
        2,   # day_of_week (Wednesday)
    ]])

    pred_clicks = 0
    pred_conversions = 0

    if 'clicks' in results:
        rf = results['clicks'].get('Random Forest', {}).get('model')
        if rf:
            pred_clicks = max(0, round(float(rf.predict(features)[0])))

    if 'conversions' in results:
        rf = results['conversions'].get('Random Forest', {}).get('model')
        if rf:
            pred_conversions = max(0, round(float(rf.predict(features)[0])))

    ctr = round(pred_clicks / impressions * 100, 4) if impressions > 0 else 0
    conv_rate = round(pred_conversions / pred_clicks * 100, 4) if pred_clicks > 0 else 0
    perf_score = min(100, round(ctr * 10 + conv_rate * 5, 2))

    return {
        'predicted_clicks': pred_clicks,
        'predicted_conversions': pred_conversions,
        'expected_ctr': ctr,
        'conversion_rate': conv_rate,
        'performance_score': perf_score,
    }

def compare_models(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        results[name] = score

    best_model = max(results, key=results.get)

    return results, best_model
def get_model_comparison(trained):
    comparison = {}

    if not trained or 'results' not in trained:
        return comparison

    for target, models in trained['results'].items():
        comparison[target] = {}
        for name, metrics in models.items():
            comparison[target][name] = {
                'r2': metrics.get('r2'),
                'mae': metrics.get('mae')
            }

    return comparison
