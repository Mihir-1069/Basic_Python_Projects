# src/train_model.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_models(df):
    """
    Trains Linear Regression, Random Forest, and XGBoost models on the data.
    Returns a dictionary of results including model predictions and scores.
    """
    # Define features (X) and target (y)
    features = ['depth'] \
               + [col for col in df.columns if col.startswith('region_')]
    X = df[features]
    y = df['mag']
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    results = {}
    
    # 1) Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    results['LinearRegression'] = {
        'model': lr,
        'predictions': pred_lr,
        'mae': mean_absolute_error(y_test, pred_lr),
        'rmse': mean_squared_error(y_test, pred_lr) ** 0.5,
        'y_test': y_test.values
    }
    print(f"Linear Regression MAE: {results['LinearRegression']['mae']:.3f}, "
          f"RMSE: {results['LinearRegression']['rmse']:.3f}")
    

    return results

if __name__ == "__main__":
    df=os.path(r'data/raw/query.csv')
    train_models(df)

