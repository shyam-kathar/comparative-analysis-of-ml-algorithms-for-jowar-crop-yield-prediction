from sklearn.tree import DecisionTreeRegressor
from common import load_data, evaluate

X_train, X_test, y_train, y_test = load_data()

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Decision Tree Performance:")
evaluate(y_test, y_pred)