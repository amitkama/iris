import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Adjust pandas display settings
pd.set_option('display.max_rows', 105)  # Set to 105 to account for headers and any extra lines

# Load the dataset
df = pd.read_csv('complete_length_features.csv')

# Preparing the data
X = df.drop('Label', axis=1)
y = df['Label'] - 1  # Adjust labels to start from 0

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train, collect feature importances, and calculate accuracy for Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save the model using joblib
joblib_file = "random_forest_model.joblib"
joblib.dump(model, joblib_file)

# Get feature importances
feature_importances = model.feature_importances_

# Sort and print first 100 feature importances for the Random Forest model
fi_sorted = pd.Series(feature_importances, index=X.columns).sort_values(ascending=False).head(100)
print("\nRandom Forest Sorted Feature Importances (Top 100):")
print(fi_sorted)

# Aggregate and sort features by average importance
fi_df = pd.DataFrame({'Random Forest': feature_importances}, index=X.columns)
fi_df['mean_importance'] = fi_df.mean(axis=1)
sorted_features = fi_df.sort_values(by='mean_importance', ascending=False).head(100)

print("\nSorted Features by Average Importance (Top 100):")
print(sorted_features)

# Print final accuracy result for Random Forest model
print("\nFinal Accuracy Result:")
print(f"Random Forest: {accuracy:.4f}")

print(f"\nModel saved as {joblib_file}")