import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))
    
    def train(self, patterns):
        for pattern in patterns:
            pattern = pattern.reshape(self.n_neurons, 1)
            self.weights += np.dot(pattern, pattern.T)
        np.fill_diagonal(self.weights, 0)
    
    def update(self, pattern):
        pattern = pattern.copy()
        for _ in range(self.n_neurons):
            for i in range(self.n_neurons):
                s = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if s >= 0 else -1
        return pattern

    def predict(self, pattern, steps=5):
        for _ in range(steps):
            pattern = self.update(pattern)
        return pattern

class HopfieldManager:
    def __init__(self, patterns):
        self.patterns = patterns
        self.network = HopfieldNetwork(len(patterns[0]))
    
    def train_network(self):
        self.network.train(self.patterns)
    
    def test_pattern(self, pattern):
        return self.network.predict(pattern)

# Función para generar un patrón de aro
def generate_ring_pattern(size):
    pattern = np.ones((size, size))
    inner_radius = size // 4
    outer_radius = size // 2
    center = size // 2
    for i in range(size):
        for j in range(size):
            distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            if distance < outer_radius and distance >= inner_radius:
                pattern[i, j] = -1
    return pattern.flatten()

if __name__ == "__main__":
    # Definir el tamaño de la matriz y generar un patrón de aro
    matrix_size = 10
    ring_pattern = generate_ring_pattern(matrix_size)

    # Agregar ruido al patrón del aro
    noisy_pattern = ring_pattern.copy()
    noisy_pattern[np.random.choice(range(len(noisy_pattern)), int(len(noisy_pattern) * 0.2))] *= -1  # Añadir ruido al 20% de las neuronas

    # Crear el administrador de la red de Hopfield
    manager = HopfieldManager([ring_pattern])

    # Entrenar la red
    manager.train_network()

    # Probar la red con el patrón de entrada ruidoso
    output_pattern = manager.test_pattern(noisy_pattern)

    # Mostrar los resultados gráficamente
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(ring_pattern.reshape(matrix_size, matrix_size), cmap='binary')
    axs[0].set_title('Patrón original del aro')

    axs[1].imshow(noisy_pattern.reshape(matrix_size, matrix_size), cmap='binary')
    axs[1].set_title('Patrón de entrada ruidoso')

    axs[2].imshow(output_pattern.reshape(matrix_size, matrix_size), cmap='binary')
    axs[2].set_title('Patrón recuperado')

    plt.show()