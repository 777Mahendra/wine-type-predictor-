import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Load datasets
url_red = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
url_white = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

df_red = pd.read_csv(url_red, sep=';')
df_white = pd.read_csv(url_white, sep=';')

# Label wine types: 1 for red, 0 for white
df_red['is_red'] = 1
df_white['is_red'] = 0

# Merge datasets
df_combined = pd.concat([df_red, df_white], ignore_index=True)
df_combined.dropna(inplace=True)

# Visualization: Alcohol distribution
plt.figure(figsize=(10, 5))
plt.hist(df_combined[df_combined['is_red'] == 1]['alcohol'], bins=12, color='crimson', alpha=0.6, label='Red Wine')
plt.hist(df_combined[df_combined['is_red'] == 0]['alcohol'], bins=12, color='gold', edgecolor='black', alpha=0.5, label='White Wine')
plt.title("Alcohol Content Distribution by Wine Type")
plt.xlabel("Alcohol (% by volume)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Features and labels
features = df_combined.drop('is_red', axis=1)
labels = df_combined['is_red']

# Normalize feature data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
