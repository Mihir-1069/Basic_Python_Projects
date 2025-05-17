# main.py

import os
from src.preprocessing import load_and_preprocess_data
from src.train_model import train_models
from src.evaluate import evaluate_models


def main():
    # Paths to data
    raw_data_path = os.path.join("data", "raw", "query.csv")
    processed_data_path = os.path.join("data", "processed", "earthquake_data.csv")

    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(raw_data_path)
    # Save processed data for reference
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}\n")

    # 2. Train regression models
    print("Training machine learning models...")
    results = train_models(df)
    print("\nModels trained. Evaluating results...\n")

    # 3. Evaluate models and plot predictions
    evaluate_models(df, results)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
