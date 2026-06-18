# Define a custom neural network class from scratch
import numpy as np
import pandas as pd

class MyNN:
   # Initialize the class with input_size representing the number of input features
    def __init__(self, input_size):
        # Initialize the weights array with random values from a standard normal distribution
        self.w = np.random.randn(input_size)
        # Initialize the bias parameter to a baseline value of 0
        self.b = 0

   # Define the training fit method using training features, labels, epoch iterations, and learning rate
    def fit(self, X, y, epochs, lr):
        # Iterate through the specified number of training epochs
        for _ in range(epochs):
            # Compute current predictions for all inputs in the matrix (forward pass)
            y_hat = self.predict(X)
            
            # Calculate gradient of loss with respect to weights by dotting transposed inputs with errors (backward pass)
            dw = (1/len(y)) * np.dot(X.T, (y_hat - y))
            print(dw)
            
            # Calculate gradient of loss with respect to bias by taking the mean prediction error
            db = np.mean(y_hat - y)
            
            # Update weights by subtracting the weight gradient scaled by the learning rate
            self.w -= lr * dw
            # Update bias by subtracting the bias gradient scaled by the learning rate
            self.b -= lr * db

   # Define prediction method to estimate classification outputs
    def predict(self, X):
        # Calculate dot product of inputs and weights, add bias, and apply Sigmoid squashing function
        return 1 / (1 + np.exp(-(np.dot(X, self.w) + self.b)))


X = pd.DataFrame([[1,2,3,4], [1,2,3,4], [1,2,3,4]])
# print(X.head())

Y = [1,2,1]

dt = MyNN(4)

dt.fit(X,Y,10,0.1)