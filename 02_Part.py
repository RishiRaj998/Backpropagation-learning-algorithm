import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the Iris dataset
iris = load_iris()
patterns, labels = iris.data, iris.target
num_classes = len(np.unique(labels))

# Neural Network Parameters
eta = 0.1  # Learning rate
alpha = 0.2  # Momentum
tol = 0.001  # Tolerance
max_iter = 5000  # Maximum iterations

# Architecture: 4-6-6-3
n, h1, h2, p = 4, 6, 6, 3
Wih1 = np.random.uniform(-1, 1, (n+1, h1))
Wh1h2 = np.random.uniform(-1, 1, (h1+1, h2))
Wh2o = np.random.uniform(-1, 1, (h2+1, p))

# Initialize weight updates
DeltaWih1Old = np.zeros_like(Wih1)
DeltaWh1h2Old = np.zeros_like(Wh1h2)
DeltaWh2oOld = np.zeros_like(Wh2o)

# Input signals with bias
Si = np.hstack((np.ones((patterns.shape[0], 1)), patterns))
D = np.eye(num_classes)[labels]  # One-hot encoding

# Training loop
errors = []
sumerror = 2 * tol  # Initialize error higher than tolerance
iterations = 0
while sumerror > tol and iterations < max_iter:
    sumerror = 0
    iterations += 1
    for k in range(len(patterns)):
        # Forward pass
        Zh1 = Si[k] @ Wih1
        Sh1 = np.hstack(([1], 1 / (1 + np.exp(-Zh1))))
        Zh2 = Sh1 @ Wh1h2
        Sh2 = np.hstack(([1], 1 / (1 + np.exp(-Zh2))))
        Yj = Sh2 @ Wh2o
        Sy = 1 / (1 + np.exp(-Yj))
        
        # Compute error
        Ek = D[k] - Sy
        deltaO = Ek * Sy * (1 - Sy)
        
        # Backpropagation
        deltaH2 = (deltaO @ Wh2o.T) * Sh2 * (1 - Sh2)
        deltaH1 = (deltaH2[1:] @ Wh1h2.T) * Sh1 * (1 - Sh1)
        
        DeltaWh2o = np.outer(Sh2, deltaO)
        DeltaWh1h2 = np.outer(Sh1, deltaH2[1:])
        DeltaWih1 = np.outer(Si[k], deltaH1[1:])
        
        # Weight updates
        Wih1 += eta * DeltaWih1 + alpha * DeltaWih1Old
        Wh1h2 += eta * DeltaWh1h2 + alpha * DeltaWh1h2Old
        Wh2o += eta * DeltaWh2o + alpha * DeltaWh2oOld
        
        DeltaWih1Old = DeltaWih1
        DeltaWh1h2Old = DeltaWh1h2
        DeltaWh2oOld = DeltaWh2o
        
        sumerror += np.sum(Ek**2)
    
    errors.append(sumerror)
    print(f"Iteration {iterations}: Error = {sumerror:.6f}")

# Compute final predictions
predictions = []
for k in range(len(patterns)):
    Zh1 = Si[k] @ Wih1
    Sh1 = np.hstack(([1], 1 / (1 + np.exp(-Zh1))))
    Zh2 = Sh1 @ Wh1h2
    Sh2 = np.hstack(([1], 1 / (1 + np.exp(-Zh2))))
    Yj = Sh2 @ Wh2o
    Sy = 1 / (1 + np.exp(-Yj))
    predictions.append(np.argmax(Sy))

# Confusion Matrix
matrix = confusion_matrix(labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Plot error vs iterations
plt.plot(errors, label="Training Error")
plt.xlabel("Iterations")
plt.ylabel("Sum Squared Error")
plt.title("Error vs Iterations")
plt.legend()
plt.grid()
plt.show()

# Report results
print(f"Best Results:")
print(f"Learning Rate: {eta}, Momentum: {alpha}, Tolerance: {tol}")
print(f"Total Iterations: {iterations}")
print(f"Final Training Error: {errors[-1]:.6f}")