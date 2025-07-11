<!-- # Hearthify
This project predicts the risk of heart disease using machine learning models trained on medical data. It helps in early detection by analyzing key health indicators like age, cholesterol, and blood pressure. -->

# 🫀 Hearthify – Heart Disease Prediction Using Machine Learning

Hearthify is a full-stack web application that predicts the risk of heart disease based on user input. It uses a machine learning model trained on medical data to provide fast, accurate predictions along with confidence scores. The goal is to support early detection and raise awareness of heart-related health issues.

---

## 📌 Features

- ✅ Predicts presence or absence of heart disease
- 📊 Returns class-wise confidence scores
- 💡 Input validation with feature range checks
- 🧠 Trained with SVM and other ML models
- 🖥️ Interactive React frontend
- 🐍 Python backend with scikit-learn
- 📦 Model saved and loaded with `joblib`

---

## ⚙️ Tech Stack

### 🧠 Machine Learning (Backend)
- Python
- pandas
- scikit-learn
- matplotlib (for visualization)
- joblib (for saving model)
- Flask or FastAPI (for `/predict` endpoint)

### 💻 Frontend (React)
- React (with hooks)
- JavaScript
- Fetch API for backend communication
- Input field validation for each feature

---

## 🧠 ML Details

- **Dataset:** UCI Heart Disease Dataset
- **Target Variable:** `Heart Disease` (converted to binary)
- **Preprocessing:** StandardScaler, Stratified splitting
- **Final model used:** Support Vector Classifier (`SVC`) with `probability=True`
- **Accuracy:**
  - Logistic Regression: 87.03%
  - Decision Tree: 100%
  - Random Forest: 100%
  - SVM: 92.59%
  - KNN: 77.77%
  - Gradient Boosting: 100%

---

## 📂 Project Structure

```bash
ML_Heart_disease/
├── backend/
│   ├── model_training.ipynb        # Jupyter notebook used for training
│   ├── heart_disease_model.pkl     # Saved ML model
│   └── app.py                      # Flask or FastAPI backend
│
├── frontend/
│   ├── App.js                      # Main React app
│   ├── index.js                    # Entry point
│   └── ...                         # Other React components and files
```

---

## 🧪 Sample Inputs (Frontend)

- Age: 29-77
- Sex: 0 (Female), 1 (Male)
- Chest Pain Type: 0-3
- Resting Blood Pressure: 94-200
- Cholesterol: 126-564
- Fasting Blood Sugar > 120: 0/1
- Resting ECG Results: 0-2
- Max Heart Rate: 71-202
- Exercise Induced Angina: 0/1
- ST Depression: 0-6.2
- Slope of ST Segment: 0-2
- Major Vessels: 0-3
- Thalassemia: 1-3

---

## ▶️ Running the Project

### Backend (Flask/FastAPI)
1. Train the model using the Jupyter notebook or load the saved `.pkl` file
2. Serve predictions at an endpoint like `http://localhost:8080/predict`
3. Ensure CORS is handled if calling from frontend

### Frontend (React)
1. Run the app:

```bash
npm install
npm start
```

