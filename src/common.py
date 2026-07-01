import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data():
    df = pd.read_csv("Jower_Preprocessed_Data.csv")

    # Encoding
    df = pd.get_dummies(df, columns=['State_Name', 'Crop_Type', 'Crop'], drop_first=True)

    X = df.drop('Production_in_tons', axis=1)
    y = df['Production_in_tons']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)