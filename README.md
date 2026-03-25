# 🌱 Smart Crop & Fertilizer Recommendation System

### 📌 Overview
This project is developed as part of the **Programming for AI** course.  
The system uses **Machine Learning** to recommend the most suitable **crop and fertilizer** based on:
- Soil Nutrients (N, P, K)
- Temperature
- Rainfall
- Location

The final product is a **Flask-based web application** that provides:
- **Crop Recommendation**  
- **Fertilizer Suggestion**  
- A user-friendly **dashboard** with interactive results  

---

### ✅ Features
- **Crop Recommendation:** Predicts the best crop based on soil and climate.
- **Fertilizer Recommendation:** Suggests the most suitable fertilizer.
- **Interactive Web UI:** Built using Flask, HTML, CSS, and Bootstrap.
- **Explainable Results:** Shows reasoning for recommendations.
- **Modular Design:** Separated ML logic and web logic for easy scalability.

---

### 🛠 Tech Stack
- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Flask (Python)
- **ML Libraries:** scikit-learn, Pandas, NumPy
- **Visualization:** Matplotlib
- **Database:** SQLite (if needed)
- **Environment:** Python `venv`

---

### 📂 Project Structure
PAI/
│
├── app/
│ ├── init.py
│ ├── routes.py
│ ├── models.py
│ ├── ml/
│ │ ├── crop_model.py
│ │ └── fertilizer_model.py
│ ├── templates/
│ │ ├── base.html
│ │ ├── home.html
│ │ └── predict.html
│ └── static/
│ ├── css/
│ ├── js/
│ └── images/
│
├── instance/
│ └── pai.sqlite
│
├── tests/
│ ├── test_routes.py
│ └── test_models.py
│
├── config.py
├── run.py
├── requirements.txt
└── README.md