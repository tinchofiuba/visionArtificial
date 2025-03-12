
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

pathFotos = 'path a fotos'

def cargarFotos(path: str) -> tuple:
    fotos = []
    etiquetas = []
    clases = os.listdir(path)
    
    for clase in clases:
        path_clase = os.path.join(path, clase)
        for foto in os.listdir(path_clase):
            foto_path = os.path.join(path_clase, foto)
            img = cv2.imread(foto_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))  # Redimensionar las imágenes a un tamaño uniforme
                fotos.append(img)
                etiquetas.append(clase)
    
    return np.array(fotos), np.array(etiquetas)

def dividirFotos(fotos: np.array, etiquetas: np.array, test_size=0.2) -> tuple:
    x_train, x_test, y_train, y_test = train_test_split(fotos, etiquetas, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test

# Cargar y dividir los datos
fotos, etiquetas = cargarFotos(pathFotos)
x_train, x_test, y_train, y_test = dividirFotos(fotos, etiquetas)

# Normalizar las imágenes
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convertir etiquetas a formato numérico
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Definir la CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(np.unique(y_train)), activation='softmax')  # Número de clases
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')