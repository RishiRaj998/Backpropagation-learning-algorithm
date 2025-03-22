import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define dataset for XOR
pattern = np.array([[0, 0, 0.05],
                    [0, 1, 0.95],
                    [1, 0, 0.95],
                    [1, 1, 0.05]])

X = pattern[:, :2]
y = pattern[:, 2]

# Function to create and train a neural network
def train_nn(X, y, learning_rate=0.1, momentum=0.9, tolerance=0.001, max_epochs=10000):
    model = Sequential([
        Dense(2, input_dim=2, activation='sigmoid'),
        Dense(2, activation='sigmoid'),
        Dense(1, activation='sigmoid')
    ])

    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(loss='mse', optimizer=optimizer)

    history = model.fit(X, y, epochs=max_epochs, verbose=0, batch_size=4)
    
    # Get loss history
    loss = history.history['loss']
    
    # Find where the loss reaches below tolerance
    iterations = next((i for i, v in enumerate(loss) if v < tolerance), len(loss))
    
    # Plot error vs iterations
    plt.plot(loss)
    plt.axhline(y=tolerance, color='r', linestyle='--', label='Tolerance')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title(f'Error vs Iterations (LR={learning_rate}, Momentum={momentum})')
    plt.legend()
    plt.show()
    
    return iterations

# Experiment 1: Default parameters
iterations_default = train_nn(X, y)

# Experiment 2: Lower tolerance
iterations_tolerance = train_nn(X, y, tolerance=0.00001)

# Experiment 3: Higher learning rate and modified momentum
iterations_high_lr = train_nn(X, y, learning_rate=1.2, momentum=0.3)

# Modify dataset for AND, OR, XNOR and retrain
patterns = {
    "AND": np.array([[0, 0, 0.05], [0, 1, 0.05], [1, 0, 0.05], [1, 1, 0.95]]),
    "OR": np.array([[0, 0, 0.05], [0, 1, 0.95], [1, 0, 0.95], [1, 1, 0.95]]),
    "XNOR": np.array([[0, 0, 0.95], [0, 1, 0.05], [1, 0, 0.05], [1, 1, 0.95]])
}

tunable_params = {"learning_rate": 0.1, "momentum": 0.9, "tolerance": 0.001}

for key, value in patterns.items():
    print(f'Training for {key} classification')
    X_new, y_new = value[:, :2], value[:, 2]
    iterations = train_nn(X_new, y_new, **tunable_params)
    print(f'Iterations for {key} classification: {iterations}')