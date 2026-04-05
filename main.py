
import pandas as pd
from sklearn.model_selection import train_test_split
from model_trainer import train_and_tune_model
from data_processor import preprocess_data
from model_evaluator import evaluate_model
import joblib
import os

def run_predictive_pipeline(data_path, target_column):
    print("
--- Starting Predictive Analytics Pipeline ---")
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    print(f"Data loaded successfully. Shape: {data.shape}")
    processed_data = preprocess_data(data.copy())
    if target_column not in processed_data.columns:
        print(f"Error: Target column '{target_column}' not found in data.")
        return
    X = processed_data.drop(target_column, axis=1)
    y = processed_data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model = train_and_tune_model(X_train, y_train)
    if best_model is None:
        print("Model training failed.")
        return
    model_filename = 'predictive_model.pkl'
    joblib.dump(best_model, model_filename)
    print(f"Model saved to {model_filename}")
    evaluate_model(best_model, X_test, y_test)
    print("
--- Predictive Analytics Pipeline Finished ---")

if __name__ == "__main__":
    dummy_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9],
        'feature3': [5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 5, 5, 1, 1, 5, 5, 1, 1, 5, 5],
        'target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
    })
    dummy_data.to_csv('data.csv', index=False)
    run_predictive_pipeline('data.csv', 'target')
    if os.path.exists('data.csv'):
        os.remove('data.csv')
