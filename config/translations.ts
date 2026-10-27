export const languages = ["es", "en"] as const
export type Language = (typeof languages)[number]

export function isLanguage(value: string): value is Language {
  return (languages as readonly string[]).includes(value)
}

const es = {
  meta: {
    siteName: "Isaac Rodríguez",
    title: "Isaac Rodríguez · Portafolio",
    description:
      "Portafolio de Isaac Rodríguez: proyectos de inteligencia artificial, programación en C++ y Python, y diseño de sistemas web.",
  },
  ui: {
    skipToContent: "Saltar al contenido",
    menu: "Menú",
    lightMode: "Cambiar a modo claro",
    darkMode: "Cambiar a modo oscuro",
    switchLanguage: "Read in English",
    viewProject: "Ver proyecto",
  },
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
      loading: "Cargando el modelo…",
      canvasLabel: "Lienzo para dibujar un número",
      modelError: "No se pudo cargar el modelo. Intenta recargar la página.",
      predict: "Predecir",
      clear: "Limpiar",
    },
    flowerClassifier: {
      title: "Clasificador de Flores",
      description: "Clasificación en tiempo real usando cámara o imágenes",
      classes: ["Margarita", "Diente de león", "Rosas", "Girasoles", "Tulipanes"],
      waiting: "Esperando predicción...",
      predictionError: "Error en la predicción",
      modelError: "No se pudo cargar el modelo. Intenta recargar la página.",
      cameraError: "No se pudo acceder a la cámara.",
      selectCamera: "Seleccionar cámara",
      frontCamera: "Cámara Frontal",
      backCamera: "Cámara Trasera",
      startCamera: "Iniciar Cámara",
      switchCamera: "Cambiar Cámara",
      upload: "Subir Imagen",
      processed: "Imagen Preprocesada",
    },
    bitcoinPrediction: {
      title: "Predicción del valor de bolsa de bitcoin",
      description: "Visualización en tiempo real de las predicciones de valor de Bitcoin.",
      predictionChart: "Gráfico de Predicción",
      comparison: "Comparación Real vs. Predicción",
      realValue: "Valor Real",
      prediction: "Predicción",
      loading: "Cargando el modelo y los precios de BTC/USDT…",
      live: "Precios de cierre por minuto de BTC/USDT desde Binance. Se actualizan cada minuto.",
      unreachable:
        "No se pudo conectar con Binance (puede estar bloqueado en tu país o no hay conexión). Se muestra una serie de ejemplo generada en el navegador, no precios reales.",
      modelError: "No se pudo cargar el modelo de predicción. Intenta recargar la página.",
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
}

export type Dictionary = typeof es
export type Project = Dictionary["sections"]["ai"]["projects"][number]
export type Skill = Dictionary["sections"]["programming"]["skills"][number]

const en: Dictionary = {
  meta: {
    siteName: "Isaac Rodríguez",
    title: "Isaac Rodríguez · Portfolio",
    description:
      "Portfolio of Isaac Rodríguez: artificial intelligence projects, C++ and Python programming, and web systems design.",
  },
  ui: {
    skipToContent: "Skip to content",
    menu: "Menu",
    lightMode: "Switch to light mode",
    darkMode: "Switch to dark mode",
    switchLanguage: "Leer en español",
    viewProject: "View project",
  },
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
      loading: "Loading the model…",
      canvasLabel: "Canvas to draw a number on",
      modelError: "The model could not be loaded. Try reloading the page.",
      predict: "Predict",
      clear: "Clear",
    },
    flowerClassifier: {
      title: "Flower Classifier",
      description: "Real-time classification from the camera or an image",
      classes: ["Daisy", "Dandelion", "Roses", "Sunflowers", "Tulips"],
      waiting: "Waiting for a prediction...",
      predictionError: "Prediction failed",
      modelError: "The model could not be loaded. Try reloading the page.",
      cameraError: "The camera could not be accessed.",
      selectCamera: "Select camera",
      frontCamera: "Front camera",
      backCamera: "Back camera",
      startCamera: "Start camera",
      switchCamera: "Switch camera",
      upload: "Upload image",
      processed: "Preprocessed image",
    },
    bitcoinPrediction: {
      title: "Bitcoin Value Prediction",
      description: "Real-time visualization of Bitcoin value predictions.",
      predictionChart: "Prediction Chart",
      comparison: "Real vs. Predicted Comparison",
      realValue: "Real Value",
      prediction: "Prediction",
      loading: "Loading the model and BTC/USDT prices…",
      live: "BTC/USDT one-minute closing prices from Binance, refreshed every minute.",
      unreachable:
        "Binance could not be reached (it may be blocked in your country, or you are offline). Showing a sample series generated in the browser, not real prices.",
      modelError: "The prediction model could not be loaded. Try reloading the page.",
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
}

export const translations: Record<Language, Dictionary> = { es, en }
