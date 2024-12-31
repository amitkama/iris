import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Adjust pandas display settings
pd.set_option('display.max_rows', 115)  # Set to 105 to account for headers and any extra lines

# Load the dataset
df = pd.read_csv('complete_length_features.csv')

# Preparing the data
X = df.drop('Label', axis=1)
y = df['Label'] - 1  # Adjust labels to start from 0

# Train the model using the entire dataset
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

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

print(f"\nModel saved as {joblib_file}")