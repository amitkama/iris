import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Adjust pandas display settings
pd.set_option('display.max_rows', 105)  # Set to 105 to account for headers and any extra lines

# Load the dataset
df = pd.read_csv('complete_length_features.csv')

# Preparing the data
X = df.drop('Label', axis=1)
y = df['Label'] - 1  # Adjust labels to start from 0

# Initialize 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store results
fold_accuracies = []
feature_importances_list = []

# Perform 10-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
    print(f"\nTraining fold {fold}...")
    # Split the data for this fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Initialize the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the Random Forest model
    rf_model.fit(X_train, y_train)

    # Predict and calculate accuracy
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies.append(accuracy)

    # Collect feature importances
    feature_importances_list.append(rf_model.feature_importances_)

    # Print accuracy for the current fold
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")

# Calculate mean and standard deviation of feature importances across folds
feature_importances_df = pd.DataFrame(feature_importances_list, columns=X.columns)
mean_feature_importances = feature_importances_df.mean()
stdev_feature_importances = feature_importances_df.std()

# Calculate and display the mean and standard deviation of accuracies
mean_accuracy = np.mean(fold_accuracies)
stdev_accuracy = np.std(fold_accuracies)
print(f"\nMean Accuracy across 10 folds: {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy across 10 folds: {stdev_accuracy:.4f}")
