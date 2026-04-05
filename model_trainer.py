
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def train_and_tune_model(X_train, y_train):
    models = {
        'RandomForest': {
            'estimator': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        },
        'GradientBoosting': {
            'estimator': GradientBoostingClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        },
        'LogisticRegression': {
            'estimator': LogisticRegression(random_state=42, solver='liblinear'),
            'param_grid': {
                'C': [0.1, 1.0, 10.0]
            }
        }
    }
    best_model = None
    best_score = -1
    best_model_name = ""
    for name, config in models.items():
        grid_search = GridSearchCV(
            estimator=config['estimator'],
            param_grid=config['param_grid'],
            cv=3,
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_model_name = name
    return best_model
