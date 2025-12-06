# Intelligent-Smoke-Classification-Model-for-Fire-Detection-Using-IoT-Sensor-Data

This project implements a machine-learning model that classifies smoke events as either fire-related or non-fire (e.g., cooking, cigarettes, steam).  
I developed this system to support smarter, more reliable fire alarms by minimizing unnecessary alarm activations.

---

## 🚨 Problem Statement

Traditional smoke detectors trigger alarms based solely on smoke intensity, which leads to frequent false alarms in everyday situations. This reduces user trust and can delay responses to real fire events.

In this project, I treat the task as a **binary classification problem**:

- `Fire Alarm = 1` → fire-related smoke  
- `Fire Alarm = 0` → nuisance / non-fire smoke  

My primary objective is to **reduce false positives**, ensuring that the system triggers an alarm only when a real fire is likely.

---

## 🗂 Dataset

- **Source:** Smoke Detection Dataset (Kaggle – IoT sensor-based)  
- **Size:** ~60,000 readings (1 Hz sampling)  
- **Target column:** `Fire Alarm`  

Key features include temperature, humidity, TVOC, eCO₂, raw gas outputs (H₂, Ethanol), air pressure, and particulate matter (PM1.0).  
The dataset covers a variety of indoor and outdoor conditions, including controlled fire scenarios and nuisance smoke environments.

---

## 🧹 Preprocessing & Handling Imbalance

- Stratified 80/20 train–test split  
- Clear imbalance detected (≈ 73.6% class 1, 26.4% class 0)  
- Applied **SMOTE** to oversample the minority class and compared results with the original data  
- Scaled all features using **StandardScaler**

---

## 🤖 Model & Hyperparameter Tuning

I used **AdaBoostClassifier** with a shallow Decision Tree as the base estimator.  
AdaBoost trains weak learners sequentially, giving more weight to previously misclassified samples. This makes it effective for datasets where fire and non-fire smoke patterns overlap.

Hyperparameter tuning was performed using **RandomizedSearchCV**, optimizing:

- `n_estimators = 200`  
- `learning_rate = 0.01`  
- `max_depth = 1`  

Precision was used as the scoring metric to directly support the goal of reducing false alarms.

---

## 📊 Results (Tuned Model – Original Data)

- **Accuracy:** 0.9895  
- **Precision:** 1.0000  
- **Recall:** 0.9857  
- **F1-score:** 0.9928  
- **AUC:** 0.99  

The tuned model trained on the original data achieved the strongest overall performance.  
I selected this model because its **perfect precision** directly translates to **fewer false positives**, aligning with the project’s main objective of minimizing unnecessary alarm activations.

---

## 🧪 Files

- `smoke_detection_model.ipynb` — full workflow: EDA, preprocessing, modeling, tuning, evaluation

---

## 🔧 Tech Stack

- Python  
- Scikit-learn  
- Imbalanced-learn  
- Pandas / NumPy  
- Matplotlib / Seaborn  

---

## 🔮 Future Work

- Compare with Gradient Boosting, XGBoost, LightGBM  
- Threshold tuning to control FP/FN trade-offs  
- Deploy as a real-time IoT inference system  
- Analyze feature importance for sensor optimization
