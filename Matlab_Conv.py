import numpy as np

# Define XOR pattern
data = np.array([[0, 0, 0.05], [0, 1, 0.95], [1, 0, 0.95], [1, 1, 0.05]])

# Parameters
eta = 1.0  # Learning rate
alpha = 0.2  # Momentum
tol = 0.001  # Error tolerance
Q = 4  # Number of patterns
n, q, p = 2, 2, 1  # Architecture

# Initialize weights
Wih = 2 * np.random.rand(n+1, q) - 1  # Input to hidden layer weights
Whj = 2 * np.random.rand(q+1, p) - 1  # Hidden to output layer weights
DeltaWih = np.zeros((n+1, q))
DeltaWhj = np.zeros((q+1, p))
DeltaWihOld = np.zeros((n+1, q))
DeltaWhjOld = np.zeros((q+1, p))

# Input signals and desired output
Si = np.hstack((np.ones((Q, 1)), data[:, :2]))  # Adding bias term
D = data[:, 2].reshape(-1, 1)  # Desired values

sumerror = 2 * tol  # Initialize error higher than tolerance

# Training loop
while sumerror > tol:
    sumerror = 0
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
    
    print(f"Epoch Error: {sumerror:.6f}")