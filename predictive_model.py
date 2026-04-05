
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Function to perform data preprocessing
def preprocess_data(data):
    # Simple preprocessing: fill missing values with the mean
    for col in data.columns:
        if data[col].isnull().any():
            data[col].fillna(data[col].mean(), inplace=True)
    return data

# Function to train and tune the model
def train_and_tune_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Main function to run the predictive modeling pipeline
def run_pipeline(data_path, target_column):
    # Load data
    print("Loading data...")
    data = pd.read_csv(data_path)
    
    # Preprocess data
    print("Preprocessing data...")
    data = preprocess_data(data)
    
    # Define features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Split data into training and testing sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and tune the model
    print("Training and tuning model...")
    best_model = train_and_tune_model(X_train, y_train)
    
    # Save the trained model
    print("Saving model...")
    joblib.dump(best_model, 'predictive_model.pkl')
    
    # Evaluate the model
    print("Evaluating model...")
    y_pred = best_model.predict(X_test)
    
    # Print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return best_model

if __name__ == "__main__":
    # Create a dummy data.csv for demonstration
    print("Creating dummy data...")
    dummy_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4],
        'feature3': [5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 5, 5, 1, 1, 5],
        'target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    })
    dummy_data.to_csv('data.csv', index=False)
    
    # Run the pipeline
    run_pipeline('data.csv', 'target')
