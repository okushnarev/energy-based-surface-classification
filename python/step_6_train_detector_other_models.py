from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    input_path = Path('data/detection/train')
    export_path = Path('ml_models/')
    export_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path / 'dataset_v1.csv')
    df['n_std'] = df['dKe'] / df['std_surf']
    df = df.drop(['n_alpha', ], axis=1)

    X = df.drop('is_new', axis=1)
    y = df['is_new']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8,
                                                        random_state=69)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      train_size=0.875,
                                                      random_state=69)

    # Decision Tree
    print('Model: Decision Tree\n')
    param_grid = {
        'criterion':         ['gini', 'entropy'],
        'max_depth':         [10, 20, 40, 50, 100, 300, 1000],
        'min_samples_split': [10, 20, 40, 50, 100, 300, 1000],
        'min_samples_leaf':  [10, 20, 40, 50, 100, 300, 1000]
    }

    dt = DecisionTreeClassifier(random_state=69)

    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_val, y_val)

    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")

    best_dt = DecisionTreeClassifier(**best_params, random_state=69)
    best_dt.fit(X_train, y_train)

    y_pred = best_dt.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy:.4f}')
    print('Classification Report:\n', classification_report(y_test, y_pred))

    joblib.dump(best_dt, export_path / 'decision_tree.joblib')
    print(f'Best model saved to {export_path}')

    # Random Forest
    print('Model: Random Forest\n')
    param_grid = {
        'n_estimators':      [10, 30, 50, 100],
        'max_depth':         [5, 10, 30, 50, 100, ],
        'min_samples_split': [20, 50, 100, 300, ],
        'min_samples_leaf':  [20, 50, 100, 300, ]
    }

    rf = RandomForestClassifier()

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_val, y_val)

    best_params = grid_search.best_params_
    print('Best parameters found:', best_params)

    best_rf = RandomForestClassifier(**best_params, n_jobs=-1)
    best_rf.fit(X_train, y_train)

    y_pred = best_rf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print('Test Accuracy:', accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(best_rf, export_path / 'random_forest.joblib')
    print(f'Best model saved to {export_path}')

    # CatBoost
    print('Model: CatBoost\n')
    train_pool = Pool(X_train, y_train)
    cat_model = CatBoostClassifier()

    cat_model.fit(train_pool)
    cat_model.score(X_test, y_test)
    cat_model.save_model(export_path / 'catboost.cbm')
    print(f'Best model saved to {export_path}')

    # Logistic Regression
    print('Model: Logistic Regression\n')
    model = LogisticRegression(random_state=69)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    joblib.dump(model, export_path / 'logreg.joblib')
    print(f'Best model saved to {export_path}')
