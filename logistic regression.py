import random as rand  # For generating random numbers
import matplotlib.pyplot as plt
import numpy as np
import math

x0 = [rand.randint(160, 200) for _ in range(100)]
x1 = [rand.randint(160, 200) for _ in range(100)]
y = [rand.randint (0, 1) for _ in range(100)]

# Parameters for the model
alpha = 1e-3 # Learning rate
iterations = 1000  # Number of iterations for gradient descent
#l = 0.01  # Regularization parameter

class Model:
    def __init__(self):
        # Initialize model parameters
        self.a = rand.uniform(-0.2, 0.2)
        self.a0 = rand.uniform(-0.2, 0.2)
        self.a1 = rand.uniform(-0.2, 0.2)

    def predict(self, x0, x1):
        z = self.a + self.a0 * x0 + self.a1 * x1
        return  1 / (1 + math.exp(-z)) # Probability prediction

    def error(self, x0, x1, y):
        epsilon = 1e-15  # Small value to prevent log(0)
        return -sum(
            yi * math.log(max(self.predict(x0i, x1i), epsilon)) +
            (1 - yi) * math.log(max(1 - self.predict(x0i, x1i), epsilon))
            for x0i, x1i, yi in zip(x0, x1, y)
        ) / len(y)

    def fit_curve(self, x0, x1, y):
            # Calculate gradients
            gradient_a = sum((self.predict(x0i, x1i) - yi) for x0i, x1i, yi in zip(x0, x1, y)) / len(y)
            gradient_a0 = sum((self.predict(x0i, x1i) - yi) * x0i for x0i, x1i, yi in zip(x0, x1, y)) / len(y)
            gradient_a1 = sum((self.predict(x0i, x1i) - yi) * x1i for x0i, x1i, yi in zip(x0, x1, y)) / len(y)

            # Update parameters
            self.a -= alpha * gradient_a
            self.a0 -= alpha * gradient_a0
            self.a1 -= alpha * gradient_a1


# Model execution
model = Model()  # Create model instance
print('Initial Error:', model.error(x0, x1, y))  # Print initial error
for i in range(iterations):
    model.fit_curve(x0, x1, y)  # Fit the model
print('Final Error:', model.error(x0, x1, y))  # Print final error


# Plotting
x0_range = np.linspace(160, 200, 100)
predicted_probabilities = []

# For each value in x0_range, predict the probability for a fixed x1
for x0_val in x0_range:
    avg_x1 = np.mean(x1)  # Using the average of x1 for the curve
    prob = model.predict(x0_val, avg_x1)  # You can also vary x1 if you wish
    predicted_probabilities.append(prob)

# Plot actual data points with colors representing their class labels
plt.scatter(x0, x1, c=y, cmap='bwr', alpha=0.7, s=60, label='Data Points')  # Increased size and alpha for visibility

# Plot the prediction curve
plt.plot(x0_range, [p * 200 for p in predicted_probabilities], color='green', linewidth=1.5, label='Prediction Curve')  # Thinner line for the curve

plt.xlabel("Feature 1 (x0)")
plt.ylabel("Feature 2 (x1)")
plt.title("Logistic Regression Predictions")
plt.colorbar(label='Class')
plt.legend()
plt.grid()
plt.show()