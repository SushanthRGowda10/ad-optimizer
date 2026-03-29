from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class AdvancedAdModel:
    def __init__(self):
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor()
        }

    def train(self, X_train, y_train):
        for model in self.models.values():
            model.fit(X_train, y_train)

    def predict(self, X):
        results = {}
        for name, model in self.models.items():
            results[name] = model.predict(X)[0]

        best_model = max(results, key=results.get)
        return results, best_model