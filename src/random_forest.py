from sklearn.ensemble import RandomForestRegressor
from common import load_data, evaluate

X_train, X_test, y_train, y_test = load_data()

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Random Forest Performance:")
evaluate(y_test, y_pred)