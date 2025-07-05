# Predict
predictions = model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int)

# Display first 10 predictions
print("\nSample Predictions:")
for i in range(10):
    print(f"Predicted: {'Red Wine' if predicted_labels[i] == 1 else 'White Wine'}")
