import heapq
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import tensorflow as tf
from tensorflow import keras

# Tamaño de la imagen
ancho = 400
alto = 400

# Crear una imagen en blanco
imagen = Image.new("RGB", (ancho, alto), "white")

# Crear un objeto para dibujar en la imagen
dibujar = ImageDraw.Draw(imagen) 

# Dibuja un cuadro rojo en la esquina inferior izquierda
dibujar.rectangle([0, alto - 50, 50, alto], fill="red")

# Dibuja un cuadro verde en la esquina superior derecha
dibujar.rectangle([ancho - 50, 0, ancho, 50], fill="green")

# Genera cuadros negros aleatorios (mínimo 4)
num_cuadros = random.randint(10, 40)
cuadros_negros = []
for _ in range(num_cuadros):
    x1 = random.randint(0, ancho - 50)
    y1 = random.randint(0, alto - 50)
    x2 = x1 + 50
    y2 = y1 + 50
    cuadros_negros.append((x1, y1, x2, y2))
    dibujar.rectangle([x1, y1, x2, y2], fill="black")

# Mostrar la imagen antes de continuar
plt.imshow(np.array(imagen))
plt.axis('off')
plt.show()

# Definir los puntos de inicio y final de la ruta
inicio_x, inicio_y = 25, alto - 25  # Cuadro rojo en la esquina inferior izquierda
final_x, final_y = ancho - 25, 25  # Cuadro verde en la esquina superior derecha

# Definir las acciones posibles (movimientos) en forma de desplazamientos
acciones = [(1, 0), (0, 1), (-1, 0), (0, -1)]

# Función de costo heurística para el algoritmo A*
def heuristica(pos_actual, pos_final):
    return np.sqrt((pos_actual[0] - pos_final[0])**2 + (pos_actual[1] - pos_final[1])**2)

# Función para verificar si una posición está dentro de un cuadro negro
def en_cuadro_negro(pos, cuadros_negros):
    x, y = pos
    for cuadro in cuadros_negros:
        x1, y1, x2, y2 = cuadro
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

# Algoritmo A*
def astar(inicio, final, cuadros_negros):
    frontera = [(0, inicio)]
    came_from = {}
    costo_g = {inicio: 0}
    
    while frontera:
        _, actual = heapq.heappop(frontera)
        
        if actual == final:
            ruta = []
            while actual in came_from:
                ruta.insert(0, actual)
                actual = came_from[actual]
            return ruta
        
        for accion in acciones:
            x, y = actual
            siguiente = (x + accion[0], y + accion[1])
            
            if (
                0 <= siguiente[0] < ancho and
                0 <= siguiente[1] < alto and
                not en_cuadro_negro(siguiente, cuadros_negros)
            ):
                costo = costo_g[actual] + 1
                if siguiente not in costo_g or costo < costo_g[siguiente]:
                    costo_g[siguiente] = costo
                    prioridad = costo + heuristica(siguiente, final)
                    heapq.heappush(frontera, (prioridad, siguiente))
                    came_from[siguiente] = actual
    
    return None

# Encontrar la ruta utilizando A*
ruta = astar((inicio_x, inicio_y), (final_x, final_y), cuadros_negros)

# Dibujar la ruta en azul como una línea sólida
for i in range(len(ruta) - 1):
    x1, y1 = ruta[i]
    x2, y2 = ruta[i + 1]
    dibujar.line([(x1, y1), (x2, y2)], fill="blue", width=3)

# Mostrar la imagen con la ruta generada
plt.imshow(np.array(imagen))
#plt.axis('off')
#plt.show()

# Generar una representación del género ya generado como entrada a la red
genero_input = np.array(imagen) / 255.0  # Normaliza los valores de píxeles

# Preparar los datos de entrenamiento (imagen y ruta)
X_train = np.array([genero_input])
# Cambia la forma de y_train para que sea una sola muestra que contiene ambas coordenadas
y_train = np.array([ruta[1]])

# Luego, ajusta la forma de y_train para que sea bidimensional
y_train = y_train.reshape(-1, 2)

# Definir la arquitectura de la red neuronal
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(alto, ancho, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(2)  # Salida: coordenadas x e y
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar la red neuronal
model.fit(X_train, y_train, epochs=50, verbose=1) 

# Generar una predicción de la siguiente posición en la ruta
predicted_target = model.predict(X_train)

# Dibujar la ruta predicha en azul como una línea sólida
ruta_predicha = [ruta[0], tuple(predicted_target[0].astype(int))]

# Mostrar la imagen con la ruta predicha
plt.imshow(np.array(imagen))
plt.plot([ruta_predicha[0][0], ruta_predicha[1][0]], [ruta_predicha[0][1], ruta_predicha[1][1]], color='blue', linewidth=3)
plt.axis('off')
plt.show()
