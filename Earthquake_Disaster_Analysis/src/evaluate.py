# src/evaluate.py

import matplotlib.pyplot as plt
import numpy as np

def evaluate_models(df, results):
    """
    Evaluate model predictions by printing error metrics and plotting predictions vs. actual values.
    """
    for name, res in results.items():
        mae = res['mae']
        rmse = res['rmse']
        print(f"{name} Test MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    
    # Plot actual vs predicted for each model
    plt.figure(figsize=(6,6))
    for name, res in results.items():
        plt.scatter(res['y_test'], res['predictions'], alpha=0.5, label=name)
    plt.plot([df['mag'].min(), df['mag'].max()], [df['mag'].min(), df['mag'].max()], 'k--', lw=2)
    plt.xlabel('Actual Magnitude')
    plt.ylabel('Predicted Magnitude')
    plt.title('Predicted vs Actual Earthquake Magnitudes')
    plt.legend()
    plt.grid(True)
    plt.show()
