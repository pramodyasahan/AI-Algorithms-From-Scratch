import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from nn import train, forward_propagation

# Loading the MNIST dataset
print("Loading the MNIST dataset...")
X, Y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Preprocess the data
print("Preprocessing data...")
X = X / 255.0  # Normalize the pixel values to [0, 1]

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert labels from string to integer
Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

# Train the neural network
print("Training the neural network...")
epochs = 1000
learning_rate = 0.1

train(X_train, Y_train, epochs, learning_rate)

# Evaluate the model
print("Evaluating the model...")
_, _, _, A2_test = forward_propagation(X_test)
predictions = np.argmax(A2_test, axis=1)
accuracy = np.mean(predictions == Y_test) * 100
print(f"Test Accuracy: {accuracy:.2f}%")
