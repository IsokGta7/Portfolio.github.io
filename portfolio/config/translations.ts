import type { Translations } from "@/types"

export const translations: Record<"es" | "en", Translations> = {
  es: {
    nav: {
      ai: "Inteligencia Artificial",
      programming: "Programación",
      systems: "Diseño de Sistemas",
      posts: "Posts",
    },
    sections: {
      ai: {
        title: "Inteligencia Artificial",
        subtitle: "Redes Neuronales y Proyectos de IA",
        projects: [
          {
            title: "Predicción de números escritos a mano",
            description: "Redes Neuronales Convolucionales",
            link: "/number-prediction",
          },
          {
            title: "Detección de texto por visión de imágenes",
            description: "Procesamiento de Imágenes",
            link: "/text-detection",
          },
          {
            title: "Predicción del valor de bolsa de bitcoin",
            description: "Machine Learning",
            link: "/bitcoin-prediction",
          },
          {
            title: "Detección de texto y texto por visión de imágenes",
            description: "Computer Vision",
            link: "/text-vision",
          },
          {
            title: "Predicción del valor de bolsa de bitcoin",
            description: "Machine Learning",
            link: "/bitcoin-prediction",
          },
          {
            title: "Detección de objetos con el modelo YOLO",
            description: "Deep Learning",
            link: "/object-detection",
          },
          {
            title: "Reconocimiento facial con caras",
            description: "Computer Vision",
            link: "/facial-recognition",
          },
          {
            title: "Detección de objetos móviles con LSTMs RNN",
            description: "Redes Neuronales Recurrentes",
            link: "/mobile-object-detection",
          },
        ],
      },
      programming: {
        title: "Programación",
        subtitle: "Lenguajes y Tecnologías",
        skills: [
          {
            name: "C++",
            description: "Desarrollo de aplicaciones de alto rendimiento",
          },
          {
            name: "Python",
            description: "Desarrollo de aplicaciones y scripts",
          },
        ],
      },
      systems: {
        title: "Diseño de Sistemas",
        subtitle: "Frameworks y Tecnologías",
        skills: [
          {
            name: "Django",
            description: "Desarrollo web backend con Python",
          },
          {
            name: ".NET",
            description: "Desarrollo de aplicaciones empresariales",
          },
          {
            name: "React",
            description: "Desarrollo de interfaces modernas",
          },
        ],
      },
      posts: {
        title: "Posts",
        subtitle: "Artículos y Tutoriales",
      },
    },
    projects: {
      numberPrediction: {
        title: "Predicción de Números Escritos a Mano",
        description: "Dibuja un número en el lienzo y el modelo lo identificará.",
        result: "Número predicho:",
        predict: "Predecir",
        clear: "Limpiar",
        showDetails: "Mostrar detalles del proyecto",
        hideDetails: "Ocultar detalles del proyecto",
        markdownContent: `
# Modelo de Predicción de Números

## Preparación del Dataset
1. Utilizamos el dataset MNIST, que contiene 70,000 imágenes de dígitos escritos a mano.
2. El dataset se dividió en 60,000 imágenes de entrenamiento y 10,000 imágenes de prueba.
3. Cada imagen es de 28x28 píxeles, representada como un array 1D de 784 valores.

## Arquitectura del Modelo
Utilizamos una Red Neuronal Convolucional (CNN) con las siguientes capas:
- Capa convolucional (32 filtros, kernel 3x3)
- Capa de MaxPooling (tamaño de pool 2x2)
- Capa convolucional (64 filtros, kernel 3x3)
- Capa de MaxPooling (tamaño de pool 2x2)
- Capa Flatten
- Capa Densa (64 unidades, activación ReLU)
- Capa de salida (10 unidades, activación softmax)

## Proceso de Entrenamiento
1. Utilizamos el optimizador Adam con una tasa de aprendizaje de 0.001.
2. El modelo se entrenó durante 10 épocas con un tamaño de lote de 32.
3. Usamos la entropía cruzada categórica como función de pérdida.
4. Se aplicaron técnicas de aumento de datos para incrementar la diversidad de nuestro conjunto de entrenamiento.

## Resultados
- Precisión en entrenamiento: 99.8%
- Precisión en validación: 99.2%

El modelo demuestra una alta precisión en el reconocimiento de dígitos escritos a mano, haciéndolo adecuado para nuestra aplicación web.
        `,
        notebookContent: `
{
"cells": [
 {
  "cell_type": "markdown",
  "metadata": {},
  "source": [
   "# Notebook de Entrenamiento del Modelo de Predicción de Números"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
   "import tensorflow as tf
",
   "from tensorflow.keras import layers, models
",
   "from tensorflow.keras.datasets import mnist
",
   "
",
   "# Cargar y preprocesar el dataset MNIST
",
   "(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
",
   "train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
",
   "test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
   "# Definir la arquitectura del modelo
",
   "model = models.Sequential([
",
   "    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
",
   "    layers.MaxPooling2D((2, 2)),
",
   "    layers.Conv2D(64, (3, 3), activation='relu'),
",
   "    layers.MaxPooling2D((2, 2)),
",
   "    layers.Flatten(),
",
   "    layers.Dense(64, activation='relu'),
",
   "    layers.Dense(10, activation='softmax')
",
   "])
",
   "
",
   "model.compile(optimizer='adam',
",
   "              loss='sparse_categorical_crossentropy',
",
   "              metrics=['accuracy'])"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
   "# Entrenar el modelo
",
   "history = model.fit(train_images, train_labels, epochs=10, 
",
   "                    validation_split=0.1, batch_size=32)
",
   "
",
   "# Evaluar el modelo
",
   "test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
",
   "print(f'Precisión en prueba: {test_acc}')"
  ]
 }
],
"metadata": {
 "kernelspec": {
  "display_name": "Python 3",
  "language": "python",
  "name": "python3"
 },
 "language_info": {
  "codemirror_mode": {
   "name": "ipython",
   "version": 3
  },
  "file_extension": ".py",
  "mimetype": "text/x-python",
  "name": "python",
  "nbconvert_exporter": "python",
  "pygments_lexer": "ipython3",
  "version": "3.8.5"
 }
},
"nbformat": 4,
"nbformat_minor": 4
}
        `,
      },
      textDetection: {
        title: "Detección de Texto por Visión de Imágenes",
        description: "Sube una imagen y el modelo detectará el texto en ella.",
        uploadPrompt: "Sube una imagen para detectar texto",
        upload: "Subir Imagen",
        result: "Texto Detectado:",
        showDetails: "Mostrar detalles del proyecto",
        hideDetails: "Ocultar detalles del proyecto",
        markdownContent: `
# Modelo de Detección de Texto en Imágenes

## Preparación del Dataset
1. Utilizamos el dataset COCO-Text, que contiene imágenes con texto en escenas naturales.
2. El dataset se dividió en conjuntos de entrenamiento, validación y prueba.
3. Se realizó aumento de datos para mejorar la generalización del modelo.

## Arquitectura del Modelo
Utilizamos una arquitectura basada en YOLO (You Only Look Once) modificada para la detección de texto:
- Backbone: ResNet50
- Neck: Feature Pyramid Network (FPN)
- Head: Convolucional para detección de texto

## Proceso de Entrenamiento
1. Utilizamos el optimizador Adam con una tasa de aprendizaje de 0.001.
2. El modelo se entrenó durante 50 épocas con un tamaño de lote de 16.
3. Implementamos un planificador de tasa de aprendizaje para mejorar la convergencia.

## Resultados
- Precisión media (mAP): 0.85
- Recall: 0.82
- F1-score: 0.83

El modelo muestra un buen rendimiento en la detección de texto en imágenes de escenas naturales.
        `,
        notebookContent: `
{
"cells": [
 {
  "cell_type": "markdown",
  "metadata": {},
  "source": [
   "# Notebook de Entrenamiento del Modelo de Detección de Texto"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
   "import torch
",
   "import torchvision
",
   "from torch import nn
",
   "from torchvision.models.detection import FasterRCNN
",
   "from torchvision.models.detection.rpn import AnchorGenerator
",
   "
",
   "# Cargar el dataset COCO-Text
",
   "# (Código para cargar el dataset omitido por brevedad)"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
   "# Definir el modelo
",
   "backbone = torchvision.models.resnet50(pretrained=True)
",
   "backbone.out_channels = 256
",
   "
",
   "anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
",
   "                                   aspect_ratios=((0.5, 1.0, 2.0),))
",
   "
",
   "roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
",
   "                                                output_size=7,
",
   "                                                sampling_ratio=2)
",
   "
",
   "model = FasterRCNN(backbone,
",
   "                   num_classes=2,  # Fondo y texto
",
   "                   rpn_anchor_generator=anchor_generator,
",
   "                   box_roi_pool=roi_pooler)"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
   "# Entrenar el modelo
",
   "# (Código de entrenamiento omitido por brevedad)
",
   "
",
   "# Evaluar el modelo
",
   "# (Código de evaluación omitido por brevedad)
",
   "
",
   "print(f'mAP: {map_score}, Recall: {recall}, F1-score: {f1_score}')"
  ]
 }
],
"metadata": {
 "kernelspec": {
  "display_name": "Python 3",
  "language": "python",
  "name": "python3"
 },
 "language_info": {
  "codemirror_mode": {
   "name": "ipython",
   "version": 3
  },
  "file_extension": ".py",
  "mimetype": "text/x-python",
  "name": "python",
  "nbconvert_exporter": "python",
  "pygments_lexer": "ipython3",
  "version": "3.8.5"
 }
},
"nbformat": 4,
"nbformat_minor": 4
}
        `,
      },
      bitcoinPrediction: {
        title: "Predicción del Valor de Bitcoin",
        description: "Visualización en tiempo real de las predicciones de valor de Bitcoin.",
        predictionChart: "Gráfico de Predicción",
        comparison: "Comparación Real vs. Predicción",
        realValue: "Valor Real",
        prediction: "Predicción",
        showDetails: "Mostrar detalles del proyecto",
        hideDetails: "Ocultar detalles del proyecto",
        markdownContent: `
# Modelo de Predicción del Valor de Bitcoin

## Preparación de Datos
1. Utilizamos datos históricos de precios de Bitcoin de los últimos 5 años.
2. Los datos se normalizaron y se dividieron en conjuntos de entrenamiento y prueba.
3. Creamos secuencias de entrada de 60 días para predecir el precio del día siguiente.

## Arquitectura del Modelo
Implementamos un modelo de Red Neuronal Recurrente (RNN) con unidades LSTM:
- Capa LSTM (128 unidades)
- Capa Dropout (0.2)
- Capa LSTM (64 unidades)
- Capa Dropout (0.2)
- Capa Densa de salida (1 unidad)

## Proceso de Entrenamiento
1. Utilizamos el optimizador Adam con una tasa de aprendizaje de 0.001.
2. El modelo se entrenó durante 100 épocas con un tamaño de lote de 32.
3. Implementamos early stopping para prevenir el sobreajuste.

## Resultados
- Error Cuadrático Medio (MSE): 0.0023
- Error Absoluto Medio (MAE): 0.0381

El modelo muestra una buena capacidad para capturar tendencias en el precio de Bitcoin, aunque las predicciones a largo plazo deben interpretarse con cautela debido a la naturaleza volátil del mercado de criptomonedas.
        `,
        notebookContent: `
{
"cells": [
 {
  "cell_type": "markdown",
  "metadata": {},
  "source": [
   "# Notebook de Entrenamiento del Modelo de Predicción de Bitcoin"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
   "import numpy as np
",
   "import pandas as pd
",
   "from sklearn.preprocessing import MinMaxScaler
",
   "from tensorflow.keras.models import Sequential
",
   "from tensorflow.keras.layers import LSTM, Dense, Dropout
",
   "
",
   "# Cargar y preprocesar los datos
",
   "df = pd.read_csv('bitcoin_historical_data.csv')
",
   "prices = df['Close'].values.reshape(-1, 1)
",
   "
",
   "scaler = MinMaxScaler()
",
   "prices_scaled = scaler.fit_transform(prices)
",
   "
",
   "# Crear secuencias de entrada
",
   "def create_sequences(data, seq_length):
",
   "    X, y = [], []
",
   "    for i in range(len(data) - seq_length):
",
   "        X.append(data[i:i+seq_length])
",
   "        y.append(data[i+seq_length])
",
   "    return np.array(X), np.array(y)
",
   "
",
   "seq_length = 60
",
   "X, y = create_sequences(prices_scaled, seq_length)"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
   "# Definir y entrenar el modelo
",
   "model = Sequential([
",
   "    LSTM(128, return_sequences=True, input_shape=(seq_length, 1)),
",
   "    Dropout(0.2),
",
   "    LSTM(64),
",
   "    Dropout(0.2),
",
   "    Dense(1)
",
   "])
",
   "
",
   "model.compile(optimizer='adam', loss='mse')
",
   "
",
   "history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2,
",
   "                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
   "# Evaluar el modelo
",
   "test_predictions = model.predict(X_test)
",
   "test_predictions = scaler.inverse_transform(test_predictions)
",
   "y_test_orig = scaler.inverse_transform(y_test)
",
   "
",
   "mse = np.mean((test_predictions - y_test_orig)**2)
",
   "mae = np.mean(np.abs(test_predictions - y_test_orig))
",
   "
",
   "print(f'MSE: {mse}, MAE: {mae}')"
  ]
 }
],
"metadata": {
 "kernelspec": {
  "display_name": "Python 3",
  "language": "python",
  "name": "python3"
 },
 "language_info": {
  "codemirror_mode": {
   "name": "ipython",
   "version": 3
  },
  "file_extension": ".py",
  "mimetype": "text/x-python",
  "name": "python",
  "nbconvert_exporter": "python",
  "pygments_lexer": "ipython3",
  "version": "3.8.5"
 }
},
"nbformat": 4,
"nbformat_minor": 4
}
        `,
      },
      cpp: {
        title: "Proyectos en C++",
        description: "Una colección de proyectos desarrollados en C++.",
        projects: [
          {
            title: "Algoritmo de Ordenamiento Rápido",
            description: "Implementación eficiente del algoritmo QuickSort en C++.",
            sampleCode: `
#include <iostream>
#include <vector>

void quickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pivot = arr[high];
        int i = low - 1;

        for (int j = low; j <= high - 1; j++) {
            if (arr[j] < pivot) {
                i++;
                std::swap(arr[i], arr[j]);
            }
        }
        std::swap(arr[i + 1], arr[high]);

        int pi = i + 1;

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main() {
    std::vector<int> arr = {10, 7, 8, 9, 1, 5};
    quickSort(arr, 0, arr.size() - 1);
    
    for (int num : arr) {
        std::cout << num << " ";
    }
    return 0;
}
            `,
          },
          // Add more C++ projects here
        ],
      },
      python: {
        title: "Proyectos en Python",
        description: "Una colección de proyectos desarrollados en Python.",
        projects: [
          {
            title: "Análisis de Sentimientos con NLTK",
            description: "Un script de Python que realiza análisis de sentimientos utilizando la biblioteca NLTK.",
            sampleCode: `
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    
    if sentiment['compound'] >= 0.05:
        return "Positivo"
    elif sentiment['compound'] <= -0.05:
        return "Negativo"
    else:
        return "Neutral"

# Ejemplo de uso
text = "Me encanta programar en Python. Es muy divertido y poderoso."
result = analyze_sentiment(text)
print(f"El sentimiento del texto es: {result}")
            `,
          },
          // Add more Python projects here
        ],
      },
    },
    common: {
      markdownExplanation: "Explicación en Markdown",
      jupyterNotebook: "Jupyter Notebook",
    },
  },
  en: {
    nav: {
      ai: "Artificial Intelligence",
      programming: "Programming",
      systems: "System Design",
      posts: "Posts",
    },
    sections: {
      ai: {
        title: "Artificial Intelligence",
        subtitle: "Neural Networks and AI Projects",
        projects: [
          {
            title: "Handwritten Number Prediction",
            description: "Convolutional Neural Networks",
            link: "/number-prediction",
          },
          {
            title: "Text Detection through Computer Vision",
            description: "Image Processing",
            link: "/text-detection",
          },
          {
            title: "Bitcoin Stock Value Prediction",
            description: "Machine Learning",
            link: "/bitcoin-prediction",
          },
          {
            title: "Text and Image Detection through Computer Vision",
            description: "Computer Vision",
            link: "/text-vision",
          },
          {
            title: "Bitcoin Stock Value Prediction",
            description: "Machine Learning",
            link: "/bitcoin-prediction",
          },
          {
            title: "Object Detection with YOLO Model",
            description: "Deep Learning",
            link: "/object-detection",
          },
          {
            title: "Facial Recognition with Faces",
            description: "Computer Vision",
            link: "/facial-recognition",
          },
          {
            title: "Mobile Object Detection with LSTMs RNN",
            description: "Recurrent Neural Networks",
            link: "/mobile-object-detection",
          },
        ],
      },
      programming: {
        title: "Programming",
        subtitle: "Languages and Technologies",
        skills: [
          {
            name: "C++",
            description: "High-performance application development",
          },
          {
            name: "Python",
            description: "Application and script development",
          },
        ],
      },
      systems: {
        title: "System Design",
        subtitle: "Frameworks and Technologies",
        skills: [
          {
            name: "Django",
            description: "Backend web development with Python",
          },
          {
            name: ".NET",
            description: "Enterprise application development",
          },
          {
            name: "React",
            description: "Modern interface development",
          },
        ],
      },
      posts: {
        title: "Posts",
        subtitle: "Articles and Tutorials",
      },
    },
    projects: {
      numberPrediction: {
        title: "Handwritten Number Prediction",
        description: "Draw a number on the canvas and the model will identify it.",
        result: "Predicted number:",
        predict: "Predict",
        clear: "Clear",
        showDetails: "Show project details",
        hideDetails: "Hide project details",
        markdownContent: `
# Number Prediction Model

## Dataset Preparation
1. We used the MNIST dataset, which contains 70,000 images of handwritten digits.
2. The dataset was split into 60,000 training images and 10,000 test images.
3. Each image is 28x28 pixels, represented as a 1D array of 784 values.

## Model Architecture
We used a Convolutional Neural Network (CNN) with the following layers:
- Convolutional layer (32 filters, 3x3 kernel)
- MaxPooling layer (2x2 pool size)
- Convolutional layer (64 filters, 3x3 kernel)
- MaxPooling layer (2x2 pool size)
- Flatten layer
- Dense layer (64 units, ReLU activation)
- Output layer (10 units, softmax activation)

## Training Process
1. We used the Adam optimizer with a learning rate of 0.001.
2. The model was trained for 10 epochs with a batch size of 32.
3. We used categorical crossentropy as the loss function.
4. Data augmentation techniques were applied to increase the diversity of our training set.

## Results
- Training accuracy: 99.8%
- Validation accuracy: 99.2%

The model demonstrates high accuracy in recognizing handwritten digits, making it suitable for our web application.
        `,
        notebookContent: `
{
"cells": [
 {
  "cell_type": "markdown",
  "metadata": {},
  "source": [
   "# Number Prediction Model Training Notebook"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
   "import tensorflow as tf
",
   "from tensorflow.keras import layers, models
",
   "from tensorflow.keras.datasets import mnist
",
   "
",
   "# Load and preprocess the MNIST dataset
",
   "(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
",
   "train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
",
   "test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
   "# Define the model architecture
",
   "model = models.Sequential([
",
   "    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
",
   "    layers.MaxPooling2D((2, 2)),
",
   "    layers.Conv2D(64, (3, 3), activation='relu'),
",
   "    layers.MaxPooling2D((2, 2)),
",
   "    layers.Flatten(),
",
   "    layers.Dense(64, activation='relu'),
",
   "    layers.Dense(10, activation='softmax')
",
   "])
",
   "
",
   "model.compile(optimizer='adam',
",
   "              loss='sparse_categorical_crossentropy',
",
   "              metrics=['accuracy'])"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
   "# Train the model
",
   "history = model.fit(train_images, train_labels, epochs=10, 
",
   "                    validation_split=0.1, batch_size=32)
",
   "
",
   "# Evaluate the model
",
   "test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
",
   "print(f'Test accuracy: {test_acc}')"
  ]
 }
],
"metadata": {
 "kernelspec": {
  "display_name": "Python 3",
  "language": "python",
  "name": "python3"
 },
 "language_info": {
  "codemirror_mode": {
   "name": "ipython",
   "version": 3
  },
  "file_extension": ".py",
  "mimetype": "text/x-python",
  "name": "python",
  "nbconvert_exporter": "python",
  "pygments_lexer": "ipython3",
  "version": "3.8.5"
 }
},
"nbformat": 4,
"nbformat_minor": 4
}
        `,
      },
      textDetection: {
        title: "Text Detection through Computer Vision",
        description: "Upload an image and the model will detect the text in it.",
        uploadPrompt: "Upload an image to detect text",
        upload: "Upload Image",
        result: "Detected Text:",
        showDetails: "Show project details",
        hideDetails: "Hide project details",
        markdownContent: `
# Text Detection in Images Model

## Dataset Preparation
1. We used the COCO-Text dataset, which contains images with text in natural scenes.
2. The dataset was split into training, validation, and test sets.
3. Data augmentation was performed to improve model generalization.

## Model Architecture
We used a YOLO (You Only Look Once) based architecture modified for text detection:
- Backbone: ResNet50
- Neck: Feature Pyramid Network (FPN)
- Head: Convolutional for text detection

## Training Process
1. We used the Adam optimizer with a learning rate of 0.001.
2. The model was trained for 50 epochs with a batch size of 16.
3. We implemented a learning rate scheduler to improve convergence.

## Results
- Mean Average Precision (mAP): 0.85
- Recall: 0.82
- F1-score: 0.83

The model shows good performance in detecting text in natural scene images.
        `,
        notebookContent: `
{
"cells": [
 {
  "cell_type": "markdown",
  "metadata": {},
  "source": [
   "# Text Detection Model Training Notebook"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
   "import torch
",
   "import torchvision
",
   "from torch import nn
",
   "from torchvision.models.detection import FasterRCNN
",
   "from torchvision.models.detection.rpn import AnchorGenerator
",
   "
",
   "# Load COCO-Text dataset
",
   "# (Code for loading the dataset omitted for brevity)"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
   "# Define the model
",
   "backbone = torchvision.models.resnet50(pretrained=True)
",
   "backbone.out_channels = 256
",
   "
",
   "anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
",
   "                                   aspect_ratios=((0.5, 1.0, 2.0),))
",
   "
",
   "roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
",
   "                                                output_size=7,
",
   "                                                sampling_ratio=2)
",
   "
",
   "model = FasterRCNN(backbone,
",
   "                   num_classes=2,  # Background and text
",
   "                   rpn_anchor_generator=anchor_generator,
",
   "                   box_roi_pool=roi_pooler)"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
   "# Train the model
",
   "# (Training code omitted for brevity)
",
   "
",
   "# Evaluate the model
",
   "# (Evaluation code omitted for brevity)
",
   "
",
   "print(f'mAP: {map_score}, Recall: {recall}, F1-score: {f1_score}')"
  ]
 }
],
"metadata": {
 "kernelspec": {
  "display_name": "Python 3",
  "language": "python",
  "name": "python3"
 },
 "language_info": {
  "codemirror_mode": {
   "name": "ipython",
   "version": 3
  },
  "file_extension": ".py",
  "mimetype": "text/x-python",
  "name": "python",
  "nbconvert_exporter": "python",
  "pygments_lexer": "ipython3",
  "version": "3.8.5"
 }
},
"nbformat": 4,
"nbformat_minor": 4
}
        `,
      },
      bitcoinPrediction: {
        title: "Bitcoin Value Prediction",
        description: "Real-time visualization of Bitcoin value predictions.",
        predictionChart: "Prediction Chart",
        comparison: "Real vs. Predicted Comparison",
        realValue: "Real Value",
        prediction: "Prediction",
        showDetails: "Show project details",
        hideDetails: "Hide project details",
        markdownContent: `
# Bitcoin Value Prediction Model

## Data Preparation
1. We used historical Bitcoin price data from the last 5 years.
2. The data was normalized and split into training and test sets.
3. We created input sequences of 60 days to predict the next day's price.

## Model Architecture
We implemented a Recurrent Neural Network (RNN) model with LSTM units:
- LSTM layer (128 units)
- Dropout layer (0.2)
- LSTM layer (64 units)
- Dropout layer (0.2)
- Dense output layer (1 unit)

## Training Process
1. We used the Adam optimizer with a learning rate of 0.001.
2. The model was trained for 100 epochs with a batch size of 32.
3. We implemented early stopping to prevent overfitting.

## Results
- Mean Squared Error (MSE): 0.0023
- Mean Absolute Error (MAE): 0.0381

The model shows good capability in capturing trends in Bitcoin price, although long-term predictions should be interpreted cautiously due to the volatile nature of the cryptocurrency market.
        `,
        notebookContent: `
{
"cells": [
 {
  "cell_type": "markdown",
  "metadata": {},
  "source": [
   "# Bitcoin Prediction Model Training Notebook"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
   "import numpy as np
",
   "import pandas as pd
",
   "from sklearn.preprocessing import MinMaxScaler
",
   "from tensorflow.keras.models import Sequential
",
   "from tensorflow.keras.layers import LSTM, Dense, Dropout
",
   "
",
   "# Load and preprocess the data
",
   "df = pd.read_csv('bitcoin_historical_data.csv')
",
   "prices = df['Close'].values.reshape(-1, 1)
",
   "
",
   "scaler = MinMaxScaler()
",
   "prices_scaled = scaler.fit_transform(prices)
",
   "
",
   "# Create input sequences
",
   "def create_sequences(data, seq_length):
",
   "    X, y = [], []
",
   "    for i in range(len(data) - seq_length):
",
   "        X.append(data[i:i+seq_length])
",
   "        y.append(data[i+seq_length])
",
   "    return np.array(X), np.array(y)
",
   "
",
   "seq_length = 60
",
   "X, y = create_sequences(prices_scaled, seq_length)"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
   "# Define and train the model
",
   "model = Sequential([
",
   "    LSTM(128, return_sequences=True, input_shape=(seq_length, 1)),
",
   "    Dropout(0.2),
",
   "    LSTM(64),
",
   "    Dropout(0.2),
",
   "    Dense(1)
",
   "])
",
   "
",
   "model.compile(optimizer='adam', loss='mse')
",
   "
",
   "history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2,
",
   "                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
   "# Evaluate the model
",
   "test_predictions = model.predict(X_test)
",
   "test_predictions = scaler.inverse_transform(test_predictions)
",
   "y_test_orig = scaler.inverse_transform(y_test)
",
   "
",
   "mse = np.mean((test_predictions - y_test_orig)**2)
",
   "mae = np.mean(np.abs(test_predictions - y_test_orig))
",
   "
",
   "print(f'MSE: {mse}, MAE: {mae}')"
  ]
 }
],
"metadata": {
 "kernelspec": {
  "display_name": "Python 3",
  "language": "python",
  "name": "python3"
 },
 "language_info": {
  "codemirror_mode": {
   "name": "ipython",
   "version": 3
  },
  "file_extension": ".py",
  "mimetype": "text/x-python",
  "name": "python",
  "nbconvert_exporter": "python",
  "pygments_lexer": "ipython3",
  "version": "3.8.5"
 }
},
"nbformat": 4,
"nbformat_minor": 4
}
        `,
      },
      cpp: {
        title: "C++ Projects",
        description: "A collection of projects developed in C++.",
        projects: [
          {
            title: "QuickSort Algorithm",
            description: "Efficient implementation of the QuickSort algorithm in C++.",
            sampleCode: `
#include <iostream>
#include <vector>

void quickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pivot = arr[high];
        int i = low - 1;

        for (int j = low; j <= high - 1; j++) {
            if (arr[j] < pivot) {
                i++;
                std::swap(arr[i], arr[j]);
            }
        }
        std::swap(arr[i + 1], arr[high]);

        int pi = i + 1;

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main() {
    std::vector<int> arr = {10, 7, 8, 9, 1, 5};
    quickSort(arr, 0, arr.size() - 1);
    
    for (int num : arr) {
        std::cout << num << " ";
    }
    return 0;
}
            `,
          },
          // Add more C++ projects here
        ],
      },
      python: {
        title: "Python Projects",
        description: "A collection of projects developed in Python.",
        projects: [
          {
            title: "Sentiment Analysis with NLTK",
            description: "A Python script that performs sentiment analysis using the NLTK library.",
            sampleCode: `
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    
    if sentiment['compound'] >= 0.05:
        return "Positive"
    elif sentiment['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Example usage
text = "I love programming in Python. It's so fun and powerful."
result = analyze_sentiment(text)
print(f"The sentiment of the text is: {result}")
            `,
          },
          // Add more Python projects here
        ],
      },
    },
    common: {
      markdownExplanation: "Markdown Explanation",
      jupyterNotebook: "Jupyter Notebook",
    },
  },
}

