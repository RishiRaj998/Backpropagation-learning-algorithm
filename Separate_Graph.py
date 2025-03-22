import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_bpnn(pattern, eta, alpha, tol, max_iter=10000):
    Q, n_p1 = pattern.shape  # Number of patterns
    n = n_p1 - 1  # Number of input neurons (excluding target)
    q = 2         # Number of hidden neurons
    p = 1         # Number of output neurons

    # Initialize weights
    Wih = 2 * np.random.rand(n + 1, q) - 1  # Input-hidden weight matrix
    Whj = 2 * np.random.rand(q + 1, p) - 1  # Hidden-output weight matrix
    DeltaWihOld = np.zeros((n + 1, q))
    DeltaWhjOld = np.zeros((q + 1, p))

    # Prepare input and output signals
    Si = np.concatenate((np.ones((Q, 1)), pattern[:, :n]), axis=1)  # Input signals with bias
    D = pattern[:, n].reshape(Q, 1)  # Desired output
    
    Sh = np.ones((Q, q + 1))  # Hidden signals with bias
    Sy = np.zeros((Q, p))     # Output signals

    sumerror = 2 * tol  # To enter loop
    iteration = 0
    error_list = []
    
    while sumerror > tol and iteration < max_iter:
        sumerror = 0
        for k in range(Q):
            # Forward propagation
            Zh = np.dot(Si[k, :], Wih)  # Hidden activations
            Sh[k, 1:] = sigmoid(Zh)    # Hidden signals

            Yj = np.dot(Sh[k, :], Whj)  # Output activations
            Sy[k] = sigmoid(Yj)         # Output signals

            # Compute error
            Ek = D[k] - Sy[k]  # Error vector
            deltaO = Ek * Sy[k] * (1 - Sy[k])  # Delta output

            # Backpropagation
            DeltaWhj = np.outer(Sh[k, :], deltaO)  # Hidden-output weight update
            deltaH = np.dot(deltaO, Whj[1:, :].T) * Sh[k, 1:] * (1 - Sh[k, 1:])
            DeltaWih = np.outer(Si[k, :], deltaH)  # Input-hidden weight update

            # Update weights with momentum
            Wih += eta * DeltaWih + alpha * DeltaWihOld
            Whj += eta * DeltaWhj + alpha * DeltaWhjOld
            
            # Store previous changes
            DeltaWihOld = DeltaWih
            DeltaWhjOld = DeltaWhj
            
            sumerror += np.sum(Ek ** 2)
        
        error_list.append(sumerror)
        iteration += 1
    
    return iteration, error_list

def plot_error(error_list, title):
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(error_list)), error_list, label="Training Error")
    plt.xlabel("Iterations")
    plt.ylabel("Sum Squared Error")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# Define patterns for AND, OR, and XNOR classification
pattern = np.array([[0, 0, 0.1], [0, 1, 0.95], [1, 0, 0.95], [1, 1, 0.1]])

# Case 1: Default Parameters (eta=1.0, alpha=0.2, tol=0.001)
iterations_1, error_list_1 = train_bpnn(pattern, eta=1.0, alpha=0.2, tol=0.001)
plot_error(error_list_1, "Error vs Iterations (eta=1.0, alpha=0.2, tol=0.001)")
print(f"Iterations needed for convergence: {iterations_1}")

# Case 2: Lower Tolerance (tol=0.00001)
iterations_2, error_list_2 = train_bpnn(pattern, eta=1.0, alpha=0.2, tol=0.00001)
plot_error(error_list_2, "Error vs Iterations (eta=1.0, alpha=0.2, tol=0.00001)")
print(f"Iterations needed for convergence: {iterations_2}")

# Case 3: Modified Learning Rate and Momentum (eta=1.2, alpha=0.3)
iterations_3, error_list_3 = train_bpnn(pattern, eta=1.2, alpha=0.3, tol=0.001)
plot_error(error_list_3, "Error vs Iterations (eta=1.2, alpha=0.3, tol=0.001)")
print(f"Iterations needed for convergence: {iterations_3}")