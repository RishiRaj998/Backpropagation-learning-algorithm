import numpy as np
import matplotlib.pyplot as plt

# Define XOR pattern
data_xor = np.array([[0, 0, 0.05], [0, 1, 0.95], [1, 0, 0.95], [1, 1, 0.05]])

def train_nn(data, eta, alpha, tol, max_iter=10000):
    Q = data.shape[0]  # Number of patterns
    n, q, p = 2, 2, 1  # Architecture

    # Initialize weights
    Wih = 2 * np.random.rand(n+1, q) - 1  # Input to hidden layer weights
    Whj = 2 * np.random.rand(q+1, p) - 1  # Hidden to output layer weights
    DeltaWihOld = np.zeros((n+1, q))
    DeltaWhjOld = np.zeros((q+1, p))

    # Input signals and desired output
    Si = np.hstack((np.ones((Q, 1)), data[:, :2]))  # Adding bias term
    D = data[:, 2].reshape(-1, 1)  # Desired values
    
    sumerror = 2 * tol  # Initialize error higher than tolerance
    errors = []
    iterations = 0

    # Training loop
    while sumerror > tol and iterations < max_iter:
        sumerror = 0
        iterations += 1
        for k in range(Q):
            # Forward pass
            Zh = Si[k] @ Wih  # Hidden activations
            Sh = np.hstack(([1], 1 / (1 + np.exp(-Zh))))  # Hidden signals
            Yj = Sh @ Whj  # Output activations
            Sy = 1 / (1 + np.exp(-Yj))  # Output signals
            
            # Compute error
            Ek = D[k] - Sy  # Error vector
            deltaO = Ek * Sy * (1 - Sy)  # Output delta
            
            # Backpropagation
            DeltaWhj = np.outer(Sh, deltaO)  # Weight update for hidden-output
            deltaH = (deltaO @ Whj.T) * Sh * (1 - Sh)  # Hidden delta
            DeltaWih = np.outer(Si[k], deltaH[1:])  # Weight update for input-hidden
            
            # Update weights
            Wih += eta * DeltaWih + alpha * DeltaWihOld
            Whj += eta * DeltaWhj + alpha * DeltaWhjOld
            
            DeltaWihOld = DeltaWih  # Store weight changes
            DeltaWhjOld = DeltaWhj
            
            sumerror += np.sum(Ek**2)  # Compute sum squared error
        
        errors.append(sumerror)
    
    return iterations, errors

# Experiment with different settings
settings = [
    (1.0, 0.2, 0.001, "XOR, tol=0.001"),
    (1.0, 0.2, 0.00001, "XOR, tol=0.00001"),
    (1.2, 0.3, 0.001, "XOR, eta=1.2, alpha=0.3"),
]

for eta, alpha, tol, label in settings:
    iterations, errors = train_nn(data_xor, eta, alpha, tol)
    plt.plot(errors, label=f"{label}, Iter={iterations}")
    print(f"{label}: Iterations = {iterations}")

plt.xlabel("Iterations")
plt.ylabel("Sum Squared Error")
plt.title("Error vs Iterations for XOR")
plt.legend()
plt.grid()
plt.show()

# Define AND, OR, XNOR patterns
patterns = {
    "AND": np.array([[0, 0, 0.05], [0, 1, 0.05], [1, 0, 0.05], [1, 1, 0.95]]),
    "OR": np.array([[0, 0, 0.05], [0, 1, 0.95], [1, 0, 0.95], [1, 1, 0.95]]),
    "XNOR": np.array([[0, 0, 0.95], [0, 1, 0.05], [1, 0, 0.05], [1, 1, 0.95]])
}

# Train and plot for each pattern
for key, pattern in patterns.items():
    iterations, errors = train_nn(pattern, 1.0, 0.2, 0.001)
    plt.plot(errors, label=f"{key}, Iter={iterations}")
    print(f"{key}: Iterations = {iterations}")

plt.xlabel("Iterations")
plt.ylabel("Sum Squared Error")
plt.title("Error vs Iterations for AND, OR, XNOR")
plt.legend()
plt.grid()
plt.show()