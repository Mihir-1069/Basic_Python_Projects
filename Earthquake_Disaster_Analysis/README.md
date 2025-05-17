# Earthquake Disaster Data Analysis and Magnitude Prediction

This project analyzes earthquake event data and builds machine learning models to predict earthquake magnitudes.

## Project Structure

- `data/raw/query.csv`: Raw earthquake dataset (downloaded from a seismic data source).  
- `data/processed/earthquake_data.csv`: Cleaned and feature-engineered data used for modeling.  
- `notebooks/Earthquake_EDA.ipynb`: Jupyter notebook with exploratory data analysis and visualizations.  
- `src/preprocessing.py`: Script to load and preprocess the raw data (handling missing values, time features, region encoding).  
- `src/train_model.py`: Script to train regression models (Linear Regression, Random Forest, XGBoost).  
- `src/evaluate.py`: Script to evaluate trained models (printing MAE, RMSE and plotting predictions).  
- `main.py`: Main pipeline that runs preprocessing, training, and evaluation in sequence.  
- `requirements.txt`: Python dependencies.  
- `README.md`: (This file) instructions to run the project.

## Setup Instructions

1. **Install Dependencies:**  
   ```bash
   pip install -r requirements.txt
