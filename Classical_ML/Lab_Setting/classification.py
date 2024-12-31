import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Adjust pandas display settings
pd.set_option('display.max_rows', 105)  # Set to 105 to account for headers and any extra lines

# Load the dataset
df = pd.read_csv('complete_length_features.csv')

# Preparing the data
X = df.drop('Label', axis=1)
y = df['Label'] - 1  # Adjust labels to start from 0

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize a dictionary to hold feature importances and accuracies
feature_importances = {}
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

# Train, collect feature importances, and calculate accuracy
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy
    if name == 'XGBoost':
        feature_importances[name] = model.feature_importances_.tolist()
    else:
        feature_importances[name] = model.feature_importances_
    
    # Sort and print first 100 feature importances for the current model
    fi_sorted = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(100)
    print(f"\n{name} Sorted Feature Importances (Top 100):")
    print(fi_sorted)

# Aggregate and sort features by average importance
fi_df = pd.DataFrame(feature_importances, index=X.columns)
fi_df['mean_importance'] = fi_df.mean(axis=1)
sorted_features = fi_df.sort_values(by='mean_importance', ascending=False).head(100)

print("\nSorted Features by Average Importance (Top 100):")
print(sorted_features)

# Print final accuracy results for each model
print("\nFinal Accuracy Results:")
for model, accuracy in model_accuracies.items():
    print(f"{model}: {accuracy:.4f}")
