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
      "Portafolio de Isaac Rodríguez: proyectos web, de sistemas en C++ y de inteligencia artificial, con demos que corren en el navegador.",
  },
  ui: {
    skipToContent: "Saltar al contenido",
    menu: "Menú",
    lightMode: "Cambiar a modo claro",
    darkMode: "Cambiar a modo oscuro",
    switchLanguage: "Read in English",
    home: "Inicio",
  },
  nav: {
    about: "Sobre mí",
    projects: "Proyectos",
    demos: "Demos",
    contact: "Contacto",
  },
  hero: {
    role: "Desarrollo de software e inteligencia artificial",
    summary:
      "Aplicaciones web con React y Supabase, sistemas de alto rendimiento en C++ y modelos de redes neuronales que corren en el navegador.",
    viewProjects: "Ver proyectos",
  },
  about: {
    title: "Sobre mí",
    paragraphs: [
      "Programo en C++ aplicaciones de alto rendimiento con MPI y estructuras de datos, y en Python aplicaciones, scripts y procesamiento de datos.",
      "En la web he trabajado con Django para el backend, con .NET para aplicaciones empresariales y con React para interfaces modernas.",
      "En inteligencia artificial he hecho proyectos con redes neuronales convolucionales para clasificar imágenes y con redes LSTM para predecir series de tiempo; tres de esos modelos se pueden probar abajo, en Demos.",
    ],
    skillGroups: [
      {
        title: "Programación",
        skills: [
          { name: "C++", description: "Aplicaciones de alto rendimiento con MPI y estructuras de datos" },
          { name: "Python", description: "Aplicaciones, scripts y procesamiento de datos" },
        ],
      },
      {
        title: "Diseño de sistemas",
        skills: [
          { name: "Django", description: "Desarrollo web backend con Python" },
          { name: ".NET", description: "Aplicaciones empresariales" },
          { name: "React", description: "Interfaces modernas" },
        ],
      },
    ],
  },
  projects: {
    title: "Proyectos",
    subtitle: "Trabajo reciente, con el problema que resuelve y el resultado.",
    labels: {
      problem: "Problema",
      role: "Rol",
      stack: "Tecnologías",
      outcome: "Resultado",
      live: "Ver sitio",
      repo: "Código",
      privateRepo: "Repositorio privado",
    },
  },
  demos: {
    title: "Demos",
    subtitle: "Modelos de redes neuronales que corren en tu navegador con TensorFlow.js.",
    open: "Abrir demo",
    backToDemos: "Volver a las demos",
  },
  contact: {
    title: "Contacto",
    subtitle: "La forma más rápida de ver mi trabajo es mi perfil de GitHub.",
    github: "GitHub",
    linkedin: "LinkedIn",
    email: "Correo",
    cv: "Descargar CV",
  },
  footer: {
    source: "Código de este sitio",
  },
  demoPages: {
    numberPrediction: {
      title: "Predicción de números escritos a mano",
      technique: "Red neuronal convolucional",
      description: "Dibuja un número en el lienzo y el modelo lo identificará.",
      result: "Número predicho:",
      loading: "Cargando el modelo…",
      canvasLabel: "Lienzo para dibujar un número",
      modelError: "No se pudo cargar el modelo. Intenta recargar la página.",
      predict: "Predecir",
      clear: "Limpiar",
    },
    flowerClassifier: {
      title: "Clasificador de flores",
      technique: "Procesamiento de imágenes con redes neuronales",
      description: "Clasificación en tiempo real usando la cámara o una imagen.",
      classes: ["Margarita", "Diente de león", "Rosas", "Girasoles", "Tulipanes"],
      waiting: "Esperando predicción...",
      predictionError: "Error en la predicción",
      modelError: "No se pudo cargar el modelo. Intenta recargar la página.",
      cameraError: "No se pudo acceder a la cámara.",
      selectCamera: "Seleccionar cámara",
      frontCamera: "Cámara frontal",
      backCamera: "Cámara trasera",
      startCamera: "Iniciar cámara",
      switchCamera: "Cambiar cámara",
      upload: "Subir imagen",
      processed: "Imagen preprocesada",
    },
    bitcoinPrediction: {
      title: "Predicción del precio de Bitcoin",
      technique: "Red neuronal recurrente LSTM",
      description: "Predicciones del precio de BTC/USDT minuto a minuto, comparadas con el valor real.",
      predictionChart: "Gráfico de predicción",
      comparison: "Comparación real vs. predicción",
      realValue: "Valor real",
      prediction: "Predicción",
      loading: "Cargando el modelo y los precios de BTC/USDT…",
      live: "Precios de cierre por minuto de BTC/USDT desde Binance. Se actualizan cada minuto.",
      unreachable:
        "No se pudo conectar con Binance (puede estar bloqueado en tu país o no hay conexión). Se muestra una serie de ejemplo generada en el navegador, no precios reales.",
      modelError: "No se pudo cargar el modelo de predicción. Intenta recargar la página.",
    },
  },
}

export type Dictionary = typeof es
export type DemoKey = keyof Dictionary["demoPages"]

const en: Dictionary = {
  meta: {
    siteName: "Isaac Rodríguez",
    title: "Isaac Rodríguez · Portfolio",
    description:
      "Portfolio of Isaac Rodríguez: web, C++ systems and artificial intelligence projects, with demos that run in the browser.",
  },
  ui: {
    skipToContent: "Skip to content",
    menu: "Menu",
    lightMode: "Switch to light mode",
    darkMode: "Switch to dark mode",
    switchLanguage: "Leer en español",
    home: "Home",
  },
  nav: {
    about: "About",
    projects: "Projects",
    demos: "Demos",
    contact: "Contact",
  },
  hero: {
    role: "Software development and artificial intelligence",
    summary:
      "Web applications with React and Supabase, high-performance systems in C++, and neural network models that run in the browser.",
    viewProjects: "View projects",
  },
  about: {
    title: "About",
    paragraphs: [
      "I write high-performance applications in C++ using MPI and data structures, and applications, scripts and data processing in Python.",
      "On the web I have worked with Django for the backend, .NET for enterprise applications and React for modern interfaces.",
      "In artificial intelligence I have built projects with convolutional neural networks to classify images and LSTM networks to predict time series; three of those models can be tried below, under Demos.",
    ],
    skillGroups: [
      {
        title: "Programming",
        skills: [
          { name: "C++", description: "High-performance applications with MPI and data structures" },
          { name: "Python", description: "Applications, scripts and data processing" },
        ],
      },
      {
        title: "System design",
        skills: [
          { name: "Django", description: "Backend web development with Python" },
          { name: ".NET", description: "Enterprise applications" },
          { name: "React", description: "Modern interfaces" },
        ],
      },
    ],
  },
  projects: {
    title: "Projects",
    subtitle: "Recent work, with the problem it solves and the result.",
    labels: {
      problem: "Problem",
      role: "Role",
      stack: "Stack",
      outcome: "Outcome",
      live: "Visit site",
      repo: "Code",
      privateRepo: "Private repository",
    },
  },
  demos: {
    title: "Demos",
    subtitle: "Neural network models that run in your browser with TensorFlow.js.",
    open: "Open demo",
    backToDemos: "Back to the demos",
  },
  contact: {
    title: "Contact",
    subtitle: "The quickest way to see my work is my GitHub profile.",
    github: "GitHub",
    linkedin: "LinkedIn",
    email: "Email",
    cv: "Download CV",
  },
  footer: {
    source: "Source of this site",
  },
  demoPages: {
    numberPrediction: {
      title: "Handwritten number prediction",
      technique: "Convolutional neural network",
      description: "Draw a number on the canvas and the model will identify it.",
      result: "Predicted number:",
      loading: "Loading the model…",
      canvasLabel: "Canvas to draw a number on",
      modelError: "The model could not be loaded. Try reloading the page.",
      predict: "Predict",
      clear: "Clear",
    },
    flowerClassifier: {
      title: "Flower classifier",
      technique: "Image processing with neural networks",
      description: "Real-time classification from the camera or an image.",
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
      title: "Bitcoin price prediction",
      technique: "LSTM recurrent neural network",
      description: "Minute-by-minute BTC/USDT price predictions, compared with the real value.",
      predictionChart: "Prediction chart",
      comparison: "Real vs. predicted",
      realValue: "Real value",
      prediction: "Prediction",
      loading: "Loading the model and BTC/USDT prices…",
      live: "BTC/USDT one-minute closing prices from Binance, refreshed every minute.",
      unreachable:
        "Binance could not be reached (it may be blocked in your country, or you are offline). Showing a sample series generated in the browser, not real prices.",
      modelError: "The prediction model could not be loaded. Try reloading the page.",
    },
  },
}

export const translations: Record<Language, Dictionary> = { es, en }
