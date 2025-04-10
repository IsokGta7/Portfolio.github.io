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
            title: "Detección flores pormedio de imagenes",
            description: "Procesamiento de Imágenes con Redes Neuronales",
            link: "/flower-classifier",
          },
          {
            title: "Predicción del valor de bolsa de bitcoin",
            description: "LSTM Recurrent Neural Network",
            link: "/bitcoin-prediction",
          },
          {
            title: "Clasificacion de animales (Perros o Gatos)",
            description: "Procesamiento de Imágenes con Redes Neuronales",
            link: "/animal-classifier",
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
            title: "Deteccion de fraude en correos electronicos",
            description: "LSTM Recurrent Neural Networks",
            link: "/fraud-detection",
          },
        ],
      },
      programming: {
        title: "Programación",
        subtitle: "Lenguajes y Tecnologías",
        skills: [
          {
            name: "C++",
            description: "Desarrollo de aplicaciones de alto rendimiento con MPICC y estructuras de datos",
          },
          {
            name: "Python",
            description: "Desarrollo de aplicaciones, scripts y procesamiento de datos",
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
        markdownContent: ``,
        notebookContent: ``,
      },
      textDetection: {
        title: "Detección flores pormedio de imagenes",
        description: "Sube una imagen de una flor o utiliza la camara.",
        uploadPrompt: "Sube una imagen o activa la camara",
        upload: "Subir Imagen",
        result: "Flor Detectada:",
        showDetails: "Mostrar detalles del proyecto",
        hideDetails: "Ocultar detalles del proyecto",
        markdownContent: ``,
        notebookContent: ``,
      },
      bitcoinPrediction: {
        title: "Predicción del valor de bolsa de bitcoin",
        description: "Visualización en tiempo real de las predicciones de valor de Bitcoin.",
        predictionChart: "Gráfico de Predicción",
        comparison: "Comparación Real vs. Predicción",
        realValue: "Valor Real",
        prediction: "Predicción",
        showDetails: "Mostrar detalles del proyecto",
        hideDetails: "Ocultar detalles del proyecto",
        markdownContent: ``,
        notebookContent: ``,
      },
      cpp: {
        title: "Proyectos en C++",
        description: "Una colección de proyectos desarrollados en C++.",
        projects: [
          {
            title: "Algoritmo de Ordenamiento Rápido",
            description: "Implementación eficiente del algoritmo QuickSort en C++.",
            sampleCode: ``,
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
            sampleCode: ``,
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
            title: "Flower detection classifier",
            description: "Image Processing with CNN",
            link: "/flower-classifier",
          },
          {
            title: "Bitcoin Stock Value Prediction",
            description: "LSTM Recurrent Neural Networks",
            link: "/bitcoin-prediction",
          },
          {
            title: "Cat and dog classifier",
            description: "Computer Vision CNN",
            link: "/animal-classifier",
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
            title: "Fraud detection on emails",
            description: "LSTM Recurrent Neural Networks",
            link: "/fraud-detection",
          },
        ],
      },
      programming: {
        title: "Programming",
        subtitle: "Languages and Technologies",
        skills: [
          {
            name: "C++",
            description: "High-performance application development using MPI and improved data structures",
          },
          {
            name: "Python",
            description: "Application, script development and data analysis and processing",
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
        markdownContent: ``,
        notebookContent: ``,
      },
      textDetection: {
        title: "Text Detection through Computer Vision",
        description: "Upload an image and the model will detect the text in it.",
        uploadPrompt: "Upload an image to detect text",
        upload: "Upload Image",
        result: "Detected Text:",
        showDetails: "Show project details",
        hideDetails: "Hide project details",
        markdownContent: ``,
        notebookContent: ``,
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
        markdownContent: ``,
        notebookContent: ``,
      },
      cpp: {
        title: "C++ Projects",
        description: "A collection of projects developed in C++.",
        projects: [
          {
            title: "QuickSort Algorithm",
            description: "Efficient implementation of the QuickSort algorithm in C++.",
            sampleCode: ``,
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
            sampleCode: ``,
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

