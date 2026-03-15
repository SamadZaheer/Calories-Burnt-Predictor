import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

DATA_DIR = "Calories-Burnt-Prediction copy"

df_1 = pd.read_csv(f"{DATA_DIR}/calories.csv")
df_2 = pd.read_csv(f"{DATA_DIR}/exercise.csv")

df = pd.merge(df_2, df_1, how="outer")

# Encode gender: male=0, female=1
df["Gender"] = df["Gender"].map({"male": 0, "female": 1})

# All 7 physiological/workout features — Weight and Duration are valid inputs
# the user knows at prediction time and are NOT data leakage
features = df.drop(["User_ID", "Calories"], axis=1)
target = df["Calories"].values

# Evaluate on a held-out split first so we can report honest test metrics
x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.1, random_state=22
)

scaler_eval = StandardScaler()
x_train_s = scaler_eval.fit_transform(x_train)
x_test_s  = scaler_eval.transform(x_test)

eval_model = XGBRegressor(random_state=22)
eval_model.fit(x_train_s, y_train)

train_r2 = r2_score(y_train, eval_model.predict(x_train_s))
test_r2  = r2_score(y_test,  eval_model.predict(x_test_s))
test_mae = mean_absolute_error(y_test, eval_model.predict(x_test_s))

print("── Evaluation on 90/10 split ──────────────────────────")
print(f"  Features : {list(features.columns)}")
print(f"  Train R² : {train_r2:.4f}")
print(f"  Test  R² : {test_r2:.4f}  (notebook baseline: 0.9466)")
print(f"  Test  MAE: {test_mae:.2f} kcal  (notebook baseline: 10.33)")
print("───────────────────────────────────────────────────────")

# Retrain on the FULL dataset for the saved production artifacts
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(features)

model = XGBRegressor(random_state=22)
model.fit(X_all_scaled, target)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nmodel.pkl and scaler.pkl saved (trained on full 15,000 samples).")
