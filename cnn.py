from keras import Input
# Define model with Input layer
model = Sequential([
    Input(shape=(features.shape[1],)),
    Dense(16, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
