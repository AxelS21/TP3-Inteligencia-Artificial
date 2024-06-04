import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            p = p.reshape(-1, 1)
            self.weights += p @ p.T
        np.fill_diagonal(self.weights, 0)

    def retrieve(self, pattern, steps=5):
        pattern = pattern.copy()
        for _ in range(steps):
            for i in range(self.size):
                raw_input = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if raw_input >= 0 else -1
        return pattern

    def add_noise(self, pattern, noise_level):
        noisy_pattern = pattern.copy()
        num_noisy = int(noise_level * len(pattern))
        indices = np.random.choice(len(pattern), num_noisy, replace=False)
        noisy_pattern[indices] = -noisy_pattern[indices]
        return noisy_pattern

def load_image(filepath, size):
    image = Image.open(filepath).convert('L')  # Convert to grayscale
    image = image.resize((size, size), Image.LANCZOS)
    image = np.array(image)
    image = (image > 128).astype(int) * 2 - 1  # Convert to binary (-1, 1)
    return image.flatten()

def plot_pattern(pattern, size, title):
    plt.imshow(pattern.reshape((size, size)), cmap='binary')
    plt.title(title)
    plt.show()

# Definir la ruta de la imagen y el tamaño deseado
image_path = 'images/descarga.jpeg'
image_size = 64  # Cambiar al tamaño deseado, por ejemplo, 64x64 píxeles

# Cargar y procesar la imagen
image_pattern = load_image(image_path, image_size)

# Definir patrones adicionales si es necesario (para entrenamiento)
additional_patterns = [image_pattern]  # Puedes agregar más patrones si lo deseas

# Inicializar y entrenar la red de Hopfield
hopfield_net = HopfieldNetwork(size=image_size * image_size)
hopfield_net.train(additional_patterns)

# Aplicar ruido a la imagen
noisy_image_pattern = hopfield_net.add_noise(image_pattern, 0.3)

# Recuperar la imagen
retrieved_image_pattern = hopfield_net.retrieve(noisy_image_pattern)

# Visualizar los resultados
plot_pattern(image_pattern, image_size, "Original Image")
plot_pattern(noisy_image_pattern, image_size, "Noisy Image")
plot_pattern(retrieved_image_pattern, image_size, "Retrieved Image")