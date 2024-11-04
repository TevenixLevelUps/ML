import random as rand  # For generating random numbers
import matplotlib.pyplot as plt # For plotting
import numpy as np # For generating an evenly spaced range for prediction, kinda linear algebra

# Generate random height data
x = [rand.randint(160, 200) for _ in range(100)] #random x dataset
y = [78 + (xi - 170) * 0.5 + rand.uniform(-5, 5) for xi in x] # y based on x with some noise added

# Parameters for the model
alpha = 1e-5  # Learning rate, how quickly model learns
iterations = 10  # Number of iterations for gradient descent
l = 0.01  # Regularization parameter to prevent overfitting - has little effect here

class Model:
    def __init__(self):
        # Initialize model parameters
        self.b = rand.uniform(0, 1)  # the intercept
        self.k = rand.uniform(0, 1) # the slope

    def predict(self, x):
        return self.b + self.k * x  # Linear prediction

    def error(self, x, y):
        # Calculate mean squared error with L2 regularization
        mse = sum((self.predict(xi) - yi) ** 2 for xi, yi in zip(x, y)) / (2 * len(y))
        return mse + l * (self.b ** 2 + self.k ** 2)  # Regularization term

    def fit_curve(self, x, y):
            # Calculate gradients
            gradient_b = sum((self.predict(xi) - yi) for xi, yi in zip(x, y)) / len(y)
            gradient_k = sum((self.predict(xi) - yi) * xi for xi, yi in zip(x, y)) / len(y)

            # Update parameters moving against the gradients
            self.b -= alpha * gradient_b
            self.k -= alpha * gradient_k


# Model execution
model = Model()  # Create model instance
print('Initial Error:', model.error(x, y))  # Print initial error
plt.figure() # create a new figure object
plt.scatter(x, y, label="Data")  # Scatter plot of actual data points
x_pred = np.linspace(160, 200, 100)  # Generate a smooth range for prediction
for i in range(iterations):
    print('Final Error:', model.error(x, y))  # Print final error
    model.fit_curve(x, y)  # Update model for this iteration
    y_pred = [model.predict(xi) for xi in x_pred]  # Get predictions for current model. We cannot plot the line through original x cuz it is not spaced evenly there but randomly. It is a bad mistake
    # x_pred and y_pred are that predictions we draw the lines through
    if i < iterations - 1:
        plt.plot(x_pred, y_pred, 'r--', label=f"Iteration {i + 1}", alpha=0.2 + i * 0.05) #intermediate predictions are drawn with decreasing transparency and in dashed lines
        # 1st note for python syntax: f letter is necessary cuz it shows python we want the expression in {} to be evaluated
        # 2nd note for python syntax: the alpha parameter controls the transparency and takes numbers in range from 0 to 1
    else:
        plt.plot(x_pred, y_pred, 'r-', label=f"Iteration {i + 1}", alpha=0.4 + i * 0.01) #the final prediction is bright and solid
plt.xlabel("Height")
plt.ylabel("Weight")
plt.legend()
plt.show()



