# Calories Burnt Prediction – Machine Learning Web Application

A production-style **Machine Learning powered web application** that predicts the number of calories burned during physical activity using user fitness and workout parameters.

This project is designed as a **portfolio-grade ML system**, demonstrating real-world integration of a trained regression model with a Flask backend and a modern, enterprise-inspired dark user interface.

---

## Project Overview

The application allows users to input basic health and activity-related details and instantly receive an estimated calorie burn value. The system follows a complete machine learning workflow — from data preprocessing and feature scaling to model inference and UI-level visualization.

The focus of this project is **clarity, reliability, and interview readiness**, making it suitable for demonstrations, technical discussions, and portfolio evaluation.

---

## Key Features

* Accurate calorie burn prediction using a trained regression model
* End-to-end ML pipeline integration with Flask
* Enterprise-style dark user interface
* Fully responsive layout (desktop, tablet, mobile)
* Input validation and exception handling
* Fast predictions using pre-trained model and scaler
  n- Auto-Demo mode for instant interview demonstration
* Prediction confidence visualization for interpretability

---

## Machine Learning Workflow

1. User provides fitness and workout parameters
2. Numerical features are processed using a trained `StandardScaler`
3. Preprocessed inputs are passed to the regression model
4. Model predicts calories burned
5. Result is displayed with visual confidence feedback

This structure mirrors real-world ML deployment practices rather than notebook-based predictions.

---

## Technology Stack

### Frontend

* HTML5
* CSS3 (custom dark theme)
* Responsive grid-based layout

### Backend

* Python
* Flask

### Machine Learning

* NumPy
* Scikit-learn
* Regression model
* Feature scaling using StandardScaler

---

## Project Structure

```
calories-burnt-prediction-ml/
│
├── app.py                 # Flask application
├── reg_model.pkl          # Trained regression model
├── scalar.pkl             # Feature scaler
│
├── templates/
│   └── index.html         # Application UI
│
├── static/
│   └── style.css          # Custom styling
│
└── README.md
```

---

## Running the Project Locally

### Step 1: Clone the Repository

```
https://calories-burnt-prediction-iyoc.onrender.com
```

### Step 2: Create and Activate Virtual Environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### Step 3: Install Required Dependencies

```
pip install flask numpy scikit-learn
```

### Step 4: Run the Application

```
python app.py
```

### Step 5: Open in Browser

```
74.220.48.0/24
74.220.56.0/24
```

---

## Auto-Demo Mode

The Auto-Demo feature automatically fills the form with realistic fitness inputs and generates predictions instantly.

**Purpose:**

* Enables quick live demonstrations
* Eliminates manual data entry during interviews
* Showcases the complete ML workflow in one click

This feature is especially useful for technical interviews and project presentations.

---

## Use Cases

* Fitness and health analytics platforms
* Workout tracking applications
* Machine learning portfolio demonstration
* Flask–ML integration reference
* Academic and learning projects

---

## Future Enhancements

* Model uncertainty–based confidence scoring
* Explainable AI (feature importance / SHAP integration)
* Cloud deployment (Render, Railway, or AWS)
* User prediction history and analytics dashboard
* Model versioning and experiment tracking

---

## Author

**Bachalakuri Ganesh**
Machine Learning & Python Developer

---

## Notes

This project is intended for educational, portfolio, and demonstration purposes. It emphasizes clean structure, deployment-oriented thinking, and professional presentation of machine learning systems.
