# ❤️ Heart Disease Risk Prediction using Machine Learning

An end-to-end machine learning pipeline that predicts **heart disease risk in patients** using clinical data from the UCI Cleveland Heart Disease dataset — with a real-time Gradio prediction interface.

---

## 🔍 Problem Statement

Heart disease is one of the leading causes of death globally. Early identification of high-risk patients using clinical parameters can enable timely intervention. This project builds a classification system to estimate patient-level heart disease risk.

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository — Cleveland Heart Disease Dataset
- **Records:** 303 patients, 14 clinical features
- **Target:** Binary — presence (1) or absence (0) of heart disease
- **Key Features:** Age, Sex, Chest Pain Type, Max Heart Rate, Exercise Angina, ST Depression, Cholesterol, FBS, CA, Thal

---

## ⚙️ Pipeline

```
Raw Data → Preprocessing & Feature Selection → Model Training → Stacking Ensemble → Gradio UI
```

### Notebooks (run in order)

| Notebook | Description |
|---|---|
| `Preprocessing.ipynb` | Column naming, missing value handling, feature selection, encoding, scaling |
| `LogisticRegression.ipynb` | Logistic Regression baseline with 5-fold cross validation |
| `RandomForest.ipynb` | Random Forest with GridSearchCV hyperparameter tuning |
| `AdaBoosting.ipynb` | AdaBoost with Decision Tree base estimators |
| `Stacking.ipynb` | Stacking Ensemble (LR + RF + AdaBoost → AdaBoost meta-learner) |
| `ui.ipynb` | Real-time Gradio prediction interface |

---

## 📈 Model Results

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 84.5% | 0.85 |
| Random Forest | 86.9% | 0.87 |
| AdaBoost | 90.0% | 0.90 |
| **Stacking Ensemble** | **93.3%** | **0.93** |

> **Best model:** Stacking Ensemble (LR + RF + AdaBoost → AdaBoost meta-learner)

---

## 🛠️ Tech Stack

- **ML:** Python, Scikit-learn (Logistic Regression, Random Forest, AdaBoost, StackingClassifier)
- **Data:** Pandas, NumPy, Matplotlib, Seaborn
- **Deployment:** Gradio
- **Storage:** Joblib

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/ui.ipynb
```

Run all cells — the Gradio interface launches in your browser with a shareable link.

---

## 📁 Project Structure

```
heart-disease-prediction/
├── notebooks/
│   ├── Preprocessing.ipynb
│   ├── LogisticRegression.ipynb
│   ├── RandomForest.ipynb
│   ├── AdaBoosting.ipynb
│   ├── Stacking.ipynb
│   └── ui.ipynb
├── models/
│   ├── logistic_model.pkl
│   ├── random_forest_model.pkl
│   ├── ada_boosting_model.pkl
│   ├── stacked_model.pkl
│   └── scaler.pkl
├── data/
│   ├── processed_cleveland.csv
│   ├── selected_cleveland.csv
│   └── final_cleveland.csv
├── requirements.txt
└── README.md
```

---

## 🔑 Key Techniques

- Feature selection — 12 most clinically relevant features
- One-hot encoding for categorical variables (cp, thal)
- StandardScaler normalization
- 5-fold Stratified Cross Validation
- Stacking Ensemble with AdaBoost meta-learner
- Gradio deployment for real-time risk prediction

---

## 👤 Author

**Madhu Sudhan Reddy Shevva**  
B.Tech CSE (Data Science) — CVR College of Engineering, Hyderabad  
[LinkedIn](https://linkedin.com/in/madhusudhan-shevva-070a793ab) | [GitHub](https://github.com)
