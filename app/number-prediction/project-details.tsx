"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent } from "@/components/ui/card"
import ReactMarkdown from "react-markdown"
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism"

const markdownContent = `
# Number Prediction Model Training

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
`

const notebookContent = `
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
    "import tensorflow as tf\n",
    "from tensorflow.keras import layers, models\n",
    "from tensorflow.keras.datasets import mnist\n",
    "\n",
    "# Load and preprocess the MNIST dataset\n",
    "(train_images, train_labels), (test_images, test_labels) = mnist.load_data()\n",
    "train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255\n",
    "test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define the model architecture\n",
    "model = models.Sequential([\n",
    "    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),\n",
    "    layers.MaxPooling2D((2, 2)),\n",
    "    layers.Conv2D(64, (3, 3), activation='relu'),\n",
    "    layers.MaxPooling2D((2, 2)),\n",
    "    layers.Flatten(),\n",
    "    layers.Dense(64, activation='relu'),\n",
    "    layers.Dense(10, activation='softmax')\n",
    "])\n",
    "\n",
    "model.compile(optimizer='adam',\n",
    "              loss='sparse_categorical_crossentropy',\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train the model\n",
    "history = model.fit(train_images, train_labels, epochs=10, \n",
    "                    validation_split=0.1, batch_size=32)\n",
    "\n",
    "# Evaluate the model\n",
    "test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)\n",
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
`

export function ProjectDetails() {
  const [activeTab, setActiveTab] = useState("markdown")

  return (
    <Card className="mt-8">
      <CardContent className="p-6">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="markdown">Markdown Explanation</TabsTrigger>
            <TabsTrigger value="notebook">Jupyter Notebook</TabsTrigger>
          </TabsList>
          <TabsContent value="markdown" className="mt-4">
            <div className="prose dark:prose-invert max-w-none">
              <ReactMarkdown
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || "")
                    return !inline && match ? (
                      <SyntaxHighlighter
                        {...props}
                        children={String(children).replace(/\n$/, "")}
                        style={vscDarkPlus}
                        language={match[1]}
                        PreTag="div"
                      />
                    ) : (
                      <code {...props} className={className}>
                        {children}
                      </code>
                    )
                  },
                }}
              >
                {markdownContent}
              </ReactMarkdown>
            </div>
          </TabsContent>
          <TabsContent value="notebook" className="mt-4">
            <div className="prose dark:prose-invert max-w-none">
              <ReactMarkdown
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || "")
                    return !inline && match ? (
                      <SyntaxHighlighter
                        {...props}
                        children={String(children).replace(/\n$/, "")}
                        style={vscDarkPlus}
                        language={match[1]}
                        PreTag="div"
                      />
                    ) : (
                      <code {...props} className={className}>
                        {children}
                      </code>
                    )
                  },
                }}
              >
                {`# Jupyter Notebook Content

\`\`\`json
${notebookContent}
\`\`\`
`}
              </ReactMarkdown>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

