# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_iris
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encoding categorical data
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# Standardize the feature data for neural network
scaler = StandardScaler()
X_train_nn = scaler.fit_transform(X_train)
X_test_nn = scaler.transform(X_test)


# Define the ANN model
model = Sequential([
    Dense(8, input_shape=(4,), activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile the ANN model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the ANN model
model.fit(X_train_nn, y_train, epochs=100, batch_size=8, verbose=0)

# Evaluate the ANN model
test_loss, test_accuracy_nn = model.evaluate(X_test_nn, y_test)
print(f"Test Accuracy (ANN): {test_accuracy_nn}")

# Make predictions for the ANN model
predictions = model.predict(X_test_nn)
print(predictions)

# Standardize the feature data for logistic regression
X_train_lr = scaler.fit_transform(X_train)
X_test_lr = scaler.transform(X_test)

# Create and train the logistic regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_lr, y_train)

# Evaluate the logistic regression model
test_accuracy_lr = logistic_model.score(X_test_lr, y_test)
print(f"Test Accuracy (Logistic Regression): {test_accuracy_lr}")
# Make predictions for the Logistic Regression Model
predictions = model.predict(X_test_lr)
print(predictions)
