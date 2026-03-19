# 🏃 Calories Burnt Predictor

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-0d9488?style=for-the-badge&logo=streamlit)](https://samadzaheer-calories-burnt.streamlit.app)

A machine learning web app that predicts how many kilocalories you burn during a workout session. Enter your personal stats and workout details, and the app instantly returns a prediction — no page reload needed.

**Inputs:** Gender, Age, Height, Weight, workout Duration, Heart Rate, and Body Temperature
**Model performance:** R² = 0.9990 · MAE = 1.38 kcal on a held-out test set of 1,500 samples

---

## App Screenshot

![App Screenshot](screenshot.png)

---

## Tech Stack

| Layer | Tools |
|---|---|
| App framework | Python, Streamlit |
| ML model | XGBoost |
| Preprocessing | scikit-learn (StandardScaler) |
| Data handling | pandas, numpy |
| Charts | Plotly |

---

## How It Works

- **7 input features** are collected from the user: Gender (encoded 0/1), Age, Height, Weight, Duration, Heart Rate, and Body Temperature — each chosen because it directly influences energy expenditure.
- **XGBoost** (Extreme Gradient Boosting) was trained on 15,000 labelled exercise records. It builds an ensemble of decision trees that progressively correct each other's errors, making it highly accurate on structured tabular data.
- **Feature scaling** is applied via StandardScaler before training and at prediction time, ensuring all input features are on a comparable numerical scale regardless of their original units.
- The model achieves **R² = 0.9990** (explains 99.9% of variance in calories burnt) and **MAE = 1.38 kcal** — an average error smaller than the calorie content of a single almond.

---

## Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/SamadZaheer/Calories-Burnt-Predictor.git
cd Calories-Burnt-Predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model and generate model.pkl + scaler.pkl
python train_model.py

# 4. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.
