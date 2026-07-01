import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Load data
df = pd.read_csv("Jower_Preprocessed_Data.csv")

# Encoding
df = pd.get_dummies(df, columns=['State_Name', 'Crop_Type', 'Crop'], drop_first=True)

X = df.drop('Production_in_tons', axis=1)
y = df['Production_in_tons']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluation function
def get_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2

# Models dictionary
models = {
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Store results
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae, rmse, r2 = get_metrics(y_test, y_pred)

    results.append([name, mae, rmse, r2])

# Create DataFrame
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2 Score"])

# Sort by best R2
results_df = results_df.sort_values(by="R2 Score", ascending=False)

print("\nModel Comparison:\n")
print(results_df)





import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a folder to save images if it doesn't exist
output_dir = "Model_Visualizations"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sns.set_theme(style="whitegrid")

# --- 1. Save Model Comparison Chart ---
plt.figure(figsize=(10, 6))
sns.barplot(x='R2 Score', y='Model', data=results_df, palette='viridis')
plt.title('Model Comparison: R2 Score', fontsize=14)
plt.xlim(0, 1.1)
plt.tight_layout()
plt.savefig(f"{output_dir}/1_Model_Comparison.png", dpi=300)
plt.close()

# Identify the best model for the next plots
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

# --- 2. Save Actual vs. Predicted Plot ---
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_best, alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Production')
plt.ylabel('Predicted Production')
plt.title(f'Actual vs. Predicted ({best_model_name})', fontsize=14)
plt.tight_layout()
plt.savefig(f"{output_dir}/2_Actual_vs_Predicted.png", dpi=300)
plt.close()

# --- 3. Save Residual Distribution Plot ---
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred_best
sns.histplot(residuals, kde=True, color='indianred')
plt.axvline(0, color='black', linestyle='--')
plt.title(f'Residuals Distribution ({best_model_name})', fontsize=14)
plt.xlabel('Error (Actual - Predicted)')
plt.tight_layout()
plt.savefig(f"{output_dir}/3_Residual_Distribution.png", dpi=300)
plt.close()

# --- 4. Save Feature Importance Plot ---
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(10, 8))
    importances = best_model.feature_importances_
    feature_names = X.columns
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
    
    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='magma')
    plt.title(f'Top 10 Feature Importances ({best_model_name})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/4_Feature_Importance.png", dpi=300)
    plt.close()

print(f"Success! All visualizations have been saved in the '{output_dir}' folder.")