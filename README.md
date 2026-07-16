# 🫁 Pneumonia Detection Using CNN

A Deep Learning-based web application that detects **Pneumonia** from Chest X-ray images using a **Convolutional Neural Network (CNN)**. The model classifies X-ray images into two categories:

- ✅ Normal
- 🦠 Pneumonia

The project is developed using **TensorFlow**, **Keras**, and **Python**, and can be integrated with **Streamlit** for an interactive user interface.

---

## 📌 Features

- Chest X-ray image classification
- CNN-based Deep Learning model
- Binary classification (Normal / Pneumonia)
- Image upload and prediction
- High prediction accuracy
- Easy-to-use interface
- Model saved in `.keras` format

---

## 📂 Project Structure

```
Pneumonia-Detection/
│
├── pneumonia_model.keras
│
├── app.py
├── requirements.txt
├── README.md
└── model_train.ipynb
```

---

## 🧠 Model Architecture

The CNN model consists of:

- Convolution Layers
- Max Pooling Layers
- Batch Normalization
- Dropout Layers
- Dense Layers
- Sigmoid Output Layer

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV
- Pillow
- Streamlit

---

## 📊 Dataset

The dataset contains Chest X-ray images divided into three folders.

```
Dataset/

├── Train/
│   ├── NORMAL/
│   └── PNEUMONIA/

├── Validation/
│   ├── NORMAL/
│   └── PNEUMONIA/

└── Test/
    ├── NORMAL/
    └── PNEUMONIA/

Used for CNN Model

[https://www.kaggle.com/datasets/umitka/chest-x-ray-balanced]

Classes:

- NORMAL
- PNEUMONIA
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YourUsername/Pneumonia-Detection.git
```

Move into the project

```bash
cd Pneumonia-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train_model.py
```

The trained model will be saved as

```
models/pneumonia_model.keras
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📈 Prediction Classes

| Class | Meaning |
|--------|---------|
| Normal | Healthy Chest X-ray |
| Pneumonia | Pneumonia Detected |

---

## 📷 Sample Workflow

1. Open the application.
2. Upload a Chest X-ray image.
3. The image is preprocessed automatically.
4. The CNN model predicts the class.
5. The result (Normal or Pneumonia) is displayed with confidence score.

---

## 📦 Requirements

```
tensorflow
keras
numpy
opencv-python
matplotlib
pillow
streamlit
scikit-learn
```

Install using

```bash
pip install -r requirements.txt
```

---

## 📊 Model Performance

Evaluation metrics may include:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🔮 Future Improvements

- Multi-class lung disease detection
- Grad-CAM visualization
- Mobile application
- Cloud deployment
- Real-time camera prediction

---

## 👨‍💻 Author

**Mohit Gaur**

AI & Machine Learning Student

---

## 📄 License

This project is developed for educational and research purposes.
