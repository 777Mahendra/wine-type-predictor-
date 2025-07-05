# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.33, random_state=42)
