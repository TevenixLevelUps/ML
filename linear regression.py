import random as rand  # For generating random numbers
import matplotlib.pyplot as plt
import numpy as np

# Generate random height data
x = [rand.randint(160, 200) for _ in range(100)]
y = [78 + (xi - 170) * 0.5 + rand.uniform(-5, 5) for xi in x]

# Parameters for the model
alpha = 1e-5  # Learning rate
iterations = 10  # Number of iterations for gradient descent
l = 0.01  # Regularization parameter

class Model:
    def __init__(self):
        # Initialize model parameters
        self.b0 = rand.uniform(0, 1)  # Start with a smaller range
        self.b1 = rand.uniform(0, 1)

    def predict(self, x):
        return self.b0 + self.b1 * x  # Linear prediction

    def error(self, x, y):
        # Calculate mean squared error with L2 regularization
        mse = sum((self.predict(xi) - yi) ** 2 for xi, yi in zip(x, y)) / (2 * len(y))
        return mse + l * (self.b0 ** 2 + self.b1 ** 2)  # Regularization term

    def fit_curve(self, x, y):
            # Calculate gradients
            gradient_b0 = sum((self.predict(xi) - yi) for xi, yi in zip(x, y)) / len(y)
            gradient_b1 = sum((self.predict(xi) - yi) * xi for xi, yi in zip(x, y)) / len(y)

            # Update parameters
            self.b0 -= alpha * gradient_b0
            self.b1 -= alpha * gradient_b1


# Model execution
model = Model()  # Create model instance
print('Initial Error:', model.error(x, y))  # Print initial error
for i in range(iterations):
    model.fit_curve(x, y)  # Fit the model
    print('Final Error:', model.error(x, y))  # Print final error
# Plotting
    plt.figure()
    plt.scatter(x, y, label="Data")  # Scatter plot of actual data
    x_pred = np.linspace(160, 200, 100)
    y_pred = [model.predict(xi) for xi in x_pred]  # Predicted y-values for plotting
    plt.plot(x_pred, y_pred, 'r', label="Model Prediction")  # Prediction line
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.legend()
    plt.show()


