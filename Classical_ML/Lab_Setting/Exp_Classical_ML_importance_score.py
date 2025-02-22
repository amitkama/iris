import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Adjust pandas display settings
pd.set_option('display.max_rows', 105)  # Set to 105 to account for headers and any extra lines

# Load the dataset
df = pd.read_csv('complete_length_features_a2.csv')

# Preparing the data
X = df.drop('Label', axis=1)
y = df['Label'] - 1  # Adjust labels to start from 0

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model on the full dataset
rf_model.fit(X, y)

# Predict and calculate accuracy
y_pred = rf_model.predict(X)
accuracy = accuracy_score(y, y_pred)

# Collect feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame to display feature importance scores
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display accuracy and feature importances
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nFeature Importances:")
print(feature_importance_df.to_string())
