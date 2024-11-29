import random as rand
import matplotlib.pyplot as plt
import numpy as np
import math

# Parameters for the model
alpha = 5e-3  # Learning rate for stability
iterations = 100  # Total iterations
snapshot_points = [20, 40, 60, 80, 100]  # Points at which to capture decision boundaries
lambda_l1 = 0.1  # L1 regularization factor
lambda_l2 = 0.1 # L2 regularization factor
w1 = 0.9 # log loss combined metric weight
w2 = 0.1 # accuracy combined metric weight

class Model:
    def __init__(self):
        # Initialize weights to small random values near zero
        self.a = rand.uniform(-0.1, 0.1)
        self.a0 = rand.uniform(-0.1, 0.1)
        self.a1 = rand.uniform(-0.1, 0.1)

    def predict(self, x0, x1):
        z = self.a + self.a0 * x0 + self.a1 * x1
        return 1 / (1 + math.exp(-z))

    def accuracy(self, x0, x1, y):
        predictions = [1 if self.predict(x0i, x1i) > 0.5 else 0 for x0i, x1i in zip(x0, x1)]
        correct = sum(yi == pred for yi, pred in zip(y, predictions))
        return correct / len(y)

    def log_loss(self, x0, x1, y):
        epsilon = 1e-15  # Small value to prevent log(0)
        regularization = (lambda_l1 * (abs(self.a) + abs(self.a0) + abs(self.a1)) +
                          lambda_l2 * (self.a ** 2 + self.a0 ** 2 + self.a1 ** 2))
        return -sum(yi * math.log(max(self.predict(x0i, x1i), epsilon)) +
                    (1 - yi) * math.log(max(1 - self.predict(x0i, x1i), epsilon))
                    for x0i, x1i, yi in zip(x0, x1, y)) / len(y) + regularization

    def fit_curve(self, x0, x1, y):
        gradient_a = sum((self.predict(x0i, x1i) - yi) for x0i, x1i, yi in zip(x0, x1, y)) / len(y)
        gradient_a0 = sum((self.predict(x0i, x1i) - yi) * x0i for x0i, x1i, yi in zip(x0, x1, y)) / len(y)
        gradient_a1 = sum((self.predict(x0i, x1i) - yi) * x1i for x0i, x1i, yi in zip(x0, x1, y)) / len(y)

        # Calculate gradients for accuracy (using a simple approach, could be refined)
        accuracy_gradient_a = sum((self.predict(x0i, x1i) > 0.5) - yi for x0i, x1i, yi in zip(x0, x1, y)) / len(y)
        accuracy_gradient_a0 = sum(
            ((self.predict(x0i, x1i) > 0.5) - yi) * x0i for x0i, x1i, yi in zip(x0, x1, y)) / len(y)
        accuracy_gradient_a1 = sum(
            ((self.predict(x0i, x1i) > 0.5) - yi) * x1i for x0i, x1i, yi in zip(x0, x1, y)) / len(y)

        # Combine both gradients, this allows you to adjust the importance of each metric
        combined_gradient_a = gradient_a + accuracy_gradient_a
        combined_gradient_a0 = gradient_a0 + accuracy_gradient_a0
        combined_gradient_a1 = gradient_a1 + accuracy_gradient_a1

        # Update weights
        self.a -= alpha * combined_gradient_a
        self.a0 -= alpha * combined_gradient_a0
        self.a1 -= alpha * combined_gradient_a1

def generate_data(n_points, class_ratio):
    #Random points
    x0 = [rand.uniform(-700, 700) for _ in range(n_points)]
    x1 = [rand.uniform(-700, 700) for _ in range(n_points)]

    # Threshold parameters
    coef_x0 = rand.uniform(-10, 10)
    coef_x1 = rand.uniform(-10, 10)
    offset = rand.uniform(-100, 100)

    threshold_values = [coef_x0 * xi0 + coef_x1 * xi1 + offset + rand.uniform(-5, 5) for xi0, xi1 in zip(x0, x1)]

    # The threshold
    sorted_threshold_values = sorted(threshold_values)
    threshold_index = int(n_points * (1 - class_ratio))
    threshold = sorted_threshold_values[threshold_index]

    # Classes
    y = [1 if value >= threshold else 0 for value in threshold_values]

    return x0, x1, y


# Function to normalize data
def normalize_data(x0, x1):
    x0_norm = (x0 - np.mean(x0)) / np.std(x0)
    x1_norm = (x1 - np.mean(x1)) / np.std(x1)
    return x0_norm, x1_norm

# Function to train the model and capture decision boundaries
def train_and_capture_boundaries(x0, x1, y, title):
    global patience_counter, previous_combined_metric
    model = Model()
    boundaries = []

    # Normalize data
    x0, x1 = normalize_data(x0, x1)

    # Training loop with snapshots at each interval
    for i in range(iterations):
        model.fit_curve(x0, x1, y)
        current_accuracy = model.accuracy(x0, x1, y)
        current_log_loss = model.log_loss(x0, x1, y)
        average_confidence = math.exp(-1 * current_log_loss)
        combined_metric = w1 * current_log_loss + w2 * (1 - current_accuracy)

        # Calculate and display accuracy for monitoring
        if (i + 1) in snapshot_points:
            print(f"Iteration {i + 1}: Accuracy = {current_accuracy * 100:.2f}%, Confidence = {average_confidence * 100:.4f}%")


            # Capture decision boundary at snapshot points
            x0_range = np.linspace(min(x0), max(x0), 300)
            x1_range = np.linspace(min(x1), max(x1), 300)
            xx, yy = np.meshgrid(x0_range, x1_range)
            Z = np.array([model.predict(x0_val, x1_val) for x0_val, x1_val in zip(xx.ravel(), yy.ravel())])
            boundaries.append((xx, yy, Z.reshape(xx.shape)))

    # Plotting the final results
    plt.figure(figsize=(10, 6))
    plt.scatter(x0, x1, c=y, cmap='bwr', alpha=0.7, s=60, label='Data Points')

    # Plot captured decision boundaries
    if boundaries:
        num_boundaries = len(boundaries) - 1
        for idx, (xx, yy, Z) in enumerate(boundaries[:-1]):
            if num_boundaries > 0:
                color = plt.cm.plasma(idx / num_boundaries)
            else:
                color = 'black'
            plt.contour(xx, yy, Z, levels=[0.5], colors=[color], linestyles='--',
                        linewidths=1.5, alpha=0.3 + 0.7 * (idx / num_boundaries if num_boundaries > 0 else 1))
            plt.plot([], [], color=color, linestyle='--', linewidth=1.5,
                     label=f'Iter {snapshot_points[idx]} Boundary')
        final_xx, final_yy, final_Z = boundaries[-1]
        plt.contour(final_xx, final_yy, final_Z, levels=[0.5], colors='black', linewidths=2, linestyles='-')
        plt.plot([], [], color="black", linestyle="-", linewidth=2, label='Final Boundary')

        plt.colorbar(plt.contourf(final_xx, final_yy, final_Z, levels=50, cmap='bwr', alpha=0.3),
                     label='Predicted Probability')

    plt.xlabel("Feature 1 (x0)")
    plt.ylabel("Feature 2 (x1)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


# Generate datasets with different class distributions and plot them
for ratio, title in zip([0.5, 0.7, 0.9],
                        ["50:50 Class Distribution", "70:30 Class Distribution", "90:10 Class Distribution"]):
    x0, x1, y = generate_data(100, ratio)
    print("\n")
    train_and_capture_boundaries(x0, x1, y, title)