# Intelligent-Smoke-Detection-System-for-Fire-Safety

This project builds a machine learning–based smoke classification model that helps fire safety systems decide whether a detected smoke event is likely to be caused by a real fire or by non-fire sources (e.g., cooking, cigarettes, steam).  
The goal is to support smarter fire alarms by reducing unnecessary alarm activations.

---

## 🚨 Problem Statement

Traditional smoke detectors typically trigger an alarm whenever smoke exceeds a fixed threshold, regardless of its source. This often leads to false alarms in everyday situations and can reduce user trust in alarm systems.

In this project, we formulate the task as a **binary classification problem**, where the target column is:

- `Fire Alarm` = 1 → fire-related smoke  
- `Fire Alarm` = 0 → non-fire / nuisance smoke  

Our primary objective is to **reduce false positives (false alarms)** by training a model that only raises an alarm when a true fire event is likely.

---

## 🗂 Dataset

- **Source:** Smoke Detection Dataset (Kaggle – IoT sensor-based)
- **Samples:** ~60,000 readings collected at 1 Hz
- **Type:** Tabular sensor data
- **Target column:** `Fire Alarm` (binary)

Key features include:
- `Temperature[C]` – Air temperature  
- `Humidity[%]` – Air humidity  
- `TVOC[ppb]` – Total volatile organic compounds  
- `eCO2[ppm]` – CO₂ equivalent  
- `Raw H2`, `Raw Ethanol` – raw gas sensor outputs  
- `Pressure[hPa]` – Air pressure  
- `PM1.0` and related particulate measurements  

The data was collected in multiple environments (indoor/outdoor, wood/gas fires, grills, normal conditions, high humidity, etc.), making it suitable for learning diverse smoke patterns.

---

## 🧹 Preprocessing & Handling Imbalance

1. **Train–test split:**  
   - `train_test_split` with an 80/20 split  
   - `stratify=y` to preserve the original class distribution

2. **Class imbalance:**  
   - EDA showed a clear imbalance in `Fire Alarm`  
   - Majority class ≈ 73.6%, minority ≈ 26.4%  
   - We experimented with **SMOTE** to oversample the minority class and compared models trained on:
     - original data  
     - SMOTE-balanced data  

3. **Feature scaling:**  
   - `StandardScaler` fitted on the training data  
   - Applied to train, SMOTE-train, and test sets.

---

## 🤖 Model & Hyperparameter Tuning

The main model is **AdaBoostClassifier** with a shallow **Decision Tree** as the base estimator.

AdaBoost trains multiple weak learners sequentially; each new tree focuses more on samples that were misclassified previously. Misclassified instances receive higher weights, and the final prediction is obtained through a weighted combination of all weak learners.  
This adaptive mechanism makes AdaBoost suitable for our sensor data, where fire and non-fire smoke can have overlapping patterns.

Hyperparameters were tuned using **RandomizedSearchCV**, optimizing:

- `n_estimators`  
- `learning_rate`  
- `estimator__max_depth` (depth of the base decision tree)

We used **precision** as the scoring metric to directly align with the goal of reducing false positives.

Best configuration:

- `n_estimators = 200`  
- `learning_rate = 0.01`  
- `max_depth = 1` (decision stump)

---

## 📊 Results (Original Data, Tuned Model)

On the test set (original data):

- **Accuracy:** 0.9895  
- **Precision:** 1.0000  
- **Recall:** 0.9857  
- **F1-score:** 0.9928  
- **AUC:** 0.99  

Best overall performance comes from the original data after tuning, where the model achieved the highest precision with a high F1 score.  
We selected this model because our primary objective is to reduce false alarms. Higher precision directly corresponds to fewer false positives, which aligns with the goal of ensuring that the system triggers an alarm only when a true fire event is likely.

---

## 🧪 Files

- `smoke_detection_model.ipynb` – Full notebook: EDA, preprocessing, model training, tuning, and evaluation.

---

## 🔮 Future Work

- Explore additional models (e.g., Gradient Boosting, XGBoost) for comparison.
- Calibrate decision thresholds to further control the trade-off between false positives and false negatives.
- Integrate the model into a real-time IoT pipeline for deployment on embedded devices.
- Investigate feature importance and sensor selection for hardware cost optimization.
