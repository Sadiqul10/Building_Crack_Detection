# 🏢 Building Crack Detection Using Deep Learning

A deep learning-based web application for detecting building cracks from images using **MobileNetV2** and **ResNet50**. The application is built with **TensorFlow/Keras** and deployed using **Streamlit**.

---

## 📌 Project Overview

Building crack detection is an important task in structural health monitoring. Manual inspection is time-consuming and prone to human error. This project uses transfer learning with MobileNetV2 and ResNet50 to automatically classify building images into:

- Crack
- Non_Crack

The application allows users to upload an image and receive an instant prediction.

---

## ✨ Features

- Image upload through Streamlit
- Crack and Non_Crack classification
- MobileNetV2 model
- ResNet50 model
- High classification accuracy
- Fast prediction
- Easy-to-use interface

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- OpenCV
- Pillow
- Matplotlib
- Scikit-learn

---

## 📂 Project Structure

```
Building-Crack-Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── mobilenetv2_final.keras
│   └── resnet50_final.keras
│
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── images/
│
└── notebooks/
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Building-Crack-Detection.git
```

Move into the project folder

```bash
cd Building-Crack-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```


## 📊 Model Performance

### MobileNetV2

- Training Accuracy: 99.56%
- Validation Accuracy: 99.77%
- Training Loss: 0.0421
- Validation Loss: 0.0289

### ResNet50

- Trained using transfer learning
- Evaluated using Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curve
- AUC Score
- Confusion Matrix

---

## 📷 Sample Prediction

Upload a building image and the application predicts:

- Crack
- Non_Crack

with confidence score.

---

## 🎯 Future Improvements

- Multi-class crack classification
- Crack severity estimation
- Real-time camera detection
- Mobile application deployment
- Cloud deployment

---

## 👨‍💻 Author

**Sadiqul Islam**

Master of Computer Applications (MCA)

---

## 📄 License

This project is developed for educational and research purposes.