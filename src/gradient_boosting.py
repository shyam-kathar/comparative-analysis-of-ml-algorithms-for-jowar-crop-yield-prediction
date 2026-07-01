from sklearn.ensemble import GradientBoostingRegressor
from common import load_data, evaluate

X_train, X_test, y_train, y_test = load_data()

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Gradient Boosting Performance:")
evaluate(y_test, y_pred)