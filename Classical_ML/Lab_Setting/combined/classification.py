import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Initialize an empty DataFrame to collect accuracy results
all_model_accuracies = pd.DataFrame()

for n in range(4900, 65001, 100):
    # Load the dataset
    df = pd.read_csv(f'complete_length_features{n}_cleaned.csv')

    # Preparing the data
    X = df.drop('Label', axis=1)
    y = df['Label'] - 1  # Adjust labels to start from 0

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Initialize a dictionary to hold accuracies
    model_accuracies = {}

    # List to hold models
    models = [
        ('CatBoost', CatBoostClassifier(verbose=0, random_state=42)),
        ('LightGBM', LGBMClassifier(random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('XGBoost', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')),
        ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        ('Gradient Boosting', GradientBoostingClassifier(random_state=42))
    ]

    # Train and calculate accuracy
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[name] = accuracy

    # Collect accuracy results for this iteration
    accuracies_df = pd.DataFrame(model_accuracies, index=[n])
    all_model_accuracies = pd.concat([all_model_accuracies, accuracies_df])

# Save the accuracy results into a CSV file
all_model_accuracies.to_csv('model_accuracies.csv')