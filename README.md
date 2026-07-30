# Ship Image Classification Based on Content-Based Image Retrieval (CBIR)

<img width="945" height="581" alt="image" src="https://github.com/user-attachments/assets/485fc3fa-c36f-4f41-9503-f6b275a1265a" />

## Overview

This project was completed as my undergraduate capstone project at Shanghai Maritime University.

The project develops a Content-Based Image Retrieval (CBIR) system for ship image classification and retrieval. Instead of relying on manually assigned labels, the system retrieves visually similar ship images by extracting image features and calculating feature similarity.

Both traditional handcrafted features and deep learning features were evaluated to compare their retrieval performance and improve classification accuracy.

---

## Project Objectives

- Build a Content-Based Image Retrieval (CBIR) system for ship images.
- Extract visual features from ship images using multiple feature extraction methods.
- Compare the performance of traditional and deep learning approaches.
- Improve retrieval accuracy through feature fusion.
- Provide a graphical user interface for image retrieval.

---

## Features

- Ship image retrieval
- Ship image classification
- Content-Based Image Retrieval (CBIR)
- Traditional feature extraction
- Deep learning feature extraction
- Feature fusion
- Similarity calculation
- Retrieval result visualization
- User-friendly GUI

---

## Technologies

- Python
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- SQLite
- Tkinter

---

## Feature Extraction Methods

### Traditional Features

- Color Histogram
- Histogram of Oriented Gradients (HOG)
- Gabor Features
- Edge Features
- Daisy Features

### Deep Learning Features

- VGG19
- ResNet152

### Similarity Measurement

- Euclidean Distance
- Cosine Similarity

---

## Project Workflow

1. Load ship image dataset.
2. Extract image features.
3. Store image feature vectors.
4. Input a query image.
5. Calculate feature similarity.
6. Rank similar images.
7. Display retrieval results.

---

## Project Structure

```
CBIR-Ship-Retrieval/
│
├── database/              # Ship image dataset
├── feature/               # Extracted image features
├── result/                # Retrieval results
├── src/                   # Source code
├── data.csv               # Dataset information
├── requirements.txt
└── README.md
```

---

## Experimental Results

Different feature extraction methods were compared for ship image retrieval.

The experiments indicate that:

- Deep learning features generally achieve higher retrieval accuracy.
- Traditional handcrafted features remain effective for specific ship categories.
- Feature fusion improves retrieval robustness and overall system performance.

---

## Dataset
<img width="632" height="622" alt="image" src="https://github.com/user-attachments/assets/3f15bc3c-7451-40b2-b613-9360d71c102f" />


The original ship image dataset is not included in this repository due to storage limitations.

Users may replace it with their own ship image dataset following the same folder structure.

---



## Usage

Run the main program:

```bash
python main.py
```

The graphical interface will open, allowing users to upload a query image and retrieve visually similar ship images.

---

## Future Improvements

- Expand the ship image dataset.
- Improve retrieval speed.
- Explore transformer-based image feature extraction.
- Develop a web-based retrieval platform.
- Support larger-scale image databases.

---


