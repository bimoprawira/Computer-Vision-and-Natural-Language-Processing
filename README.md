# Ch. 3: Computer Vision and Natural Language Processing

This repository includes a set of machine learning and deep learning assignments aimed at building practical skills in image processing, object detection, transfer learning, and text classification using different algorithms and frameworks, including PyTorch and YOLOv5.

---

### 1. Case 1: Image Enhancement

This notebook delves into replicating smartphone camera image enhancement algorithms to improve photo quality, especially in low-light conditions. Drawing inspiration from industry leaders like Apple and Samsung, the goal is to enhance dark images by implementing techniques that simulate smartphone camera technology.

- **Libraries**: , `Scikit-learn`, `OpenCV`, `NumPy`, `Matplotlib`
- **Objective**: Digital Image Processing

- **Key Objectives**: 
  - *Image Enhancement*: Simulate advanced image-brightening techniques inspired by modern smartphone cameras to improve the quality of images taken in low-light conditions. This will involve using Max Pooling and CLAHE (Contrast Limited Adaptive Histogram Equalization) to adjust and enhance photo quality.
  - *Pooling Techniques*: Explore different pooling methods to evaluate their impact on image processing. Techniques include Max Pooling, which highlights brighter pixels; Min Pooling, which emphasizes darker pixels; and Average Pooling, which smooths pixel values by averaging them.
  - *Model Development*: Train and assess multiple classification models, including logistic regression, decision trees, random forests, and gradient boosting. The aim is to identify the model with the highest performance, providing a benchmark for image classification tasks.
  - *Image Contrast Enhancement*: Apply CLAHE to improve contrast in dark areas of images, allowing details to stand out more clearly without excessive brightening.

---

### 2. Case 2: Transfer Learning for Handwritten Digit Classification

This notebook implements transfer learning using pre-trained Convolutional Neural Network (CNN) models to enable a robot to recognize handwritten digits from the MNIST dataset. The project simulates a scenario where transfer learning is used to expedite model development within a tight deadline, leveraging the strengths of pre-trained models on ImageNet to classify handwritten digits effectively.

- **Dataset**: MNIST Handwritten Digits (10 classes)
- **Libraries**: `PyTorch`, `Torchvision`, `Scikit-learn`
- **Objective**: Transfer Learning using CNN-based Pre-trained Models

- **Key Objectives**: 
  - *Implement Transfer Learning*: Adapt pre-trained CNN models (DenseNet121, ResNet18, and Vision Transformer) to classify handwritten digits from the MNIST dataset by modifying network layers to suit the task's requirements (10 classes instead of 1,000).
  - *Model Layer Customization*: Configure DenseNet121 by adjusting its input and output layers to accommodate MNIST’s grayscale images and fewer output classes.
  - *Hyperparameter Tuning and Model Training*: Define suitable hyperparameters and train the modified DenseNet model from scratch, observing its performance across training and validation datasets.
  - *Layer Freezing*: Explore transfer learning by freezing different parts of the DenseNet model (e.g., denseblock1 and denseblock2) and retraining it to examine how freezing affects performance, with each configuration considered a separate model variant.
  - *Model Evaluation and Comparison*: Analyze and visualize the performance of each model variant, comparing results to determine the best-performing setup. Optionally, replicate the same process for ResNet18 and Vision Transformer for further insights.

---

### 3. Case 3: Object Detection using YOLOv5

This notebook aims to demonstrate real-time object detection by utilizing a pre-trained YOLOv5 model. The model will be used to recognize and label various objects in video streams, allowing users to see bounding boxes and object labels in real-time. This is particularly useful for applications where immediate visual feedback on object recognition is needed, such as in surveillance, autonomous driving, or video analytics.

- **Dataset**: Any YouTube videos
- **Libraries**: `PyTorch`, `Numpy`, `OpenCV2`
- **Objective**: Real-time Object Detection using CNN-based Pre-trained Models

- **Key Objectives**: 
  - *Real-Time Object Detection*: Implement a YOLOv5-based system to detect and label objects in live video streams, demonstrating the ability to process and display detections in real time.
  - *Frame-by-Frame Analysis*: Apply the YOLOv5 model to analyze each video frame independently, enabling the identification of multiple objects within each frame.
  - *Video Capture and Processing*: Integrate video data from YouTube, transforming a continuous stream into individual frames ready for object detection.
  - *Performance Optimization*: Evaluate and optimize the model’s performance, focusing on maintaining high detection accuracy and a smooth frames-per-second (FPS) rate suitable for real-time applications.

---

### 4. Case 4: Disaster Tweet Classification with BERT

This notebook uses BERT (Bidirectional Encoder Representations for Transformers) to classify tweets as disaster-related or not, with the aim of supporting emergency response efforts by analyzing social media content in real-time.

- **Type**: NATURAL LANGUAGE UNDERSTANDING (NLU)
- **Dataset**: Disaster Tweets
- **Libraries**: `NLTK`, `Pandas`, `Numpy`, `Scikit-learn`, `PyTorch`, `Transformers`
- **Objective**: Text Classification with Bidirectional Encoder Representation for Transformers (BERT)

- **Key Objectives**: 
  - *Data Preprocessing*: Prepare the tweet data by cleaning it, which includes removing URLs, HTML tags, and stopwords. This step is essential to reduce noise and ensure that only meaningful text is passed to the model.

  - *Tokenization*: Convert the cleaned tweets into tokens using BERT’s tokenizer, which transforms text into a format that the BERT model can process. This involves splitting the text into subwords, adding special tokens, and creating attention masks to handle varying sequence lengths.

  - *Model Training and Evaluation*: Train the fine-tuned BERT model on the disaster tweet dataset. After training, validate its accuracy and measure performance to ensure the model can reliably classify disaster-related tweets. Performance metrics like precision, recall, and F1-score will be used to assess its effectiveness in recognizing relevant conten
---

## Running the Notebooks

1. Open [Google Colab](https://colab.research.google.com/) and upload the `.ipynb` files.
2. Or you can click on the "`Open in Colab`" button at the top of the project on GitHub.


---
