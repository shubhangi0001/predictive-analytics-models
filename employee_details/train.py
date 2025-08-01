import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

base_path = r"D:\employee_details\dashboard2"

split_dir = os.path.join(base_path, "splits")
X_train = pd.read_csv(os.path.join(split_dir, "X_train.csv"))
y_train = pd.read_csv(os.path.join(split_dir, "y_train.csv"))
X_test = pd.read_csv(os.path.join(split_dir, "X_test.csv"))
y_test = pd.read_csv(os.path.join(split_dir, "y_test.csv"))

categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), X_train.select_dtypes(include=['number']).columns.tolist())
    ]
)

model_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)))
])

# Train
print(" Training model with preprocessing...")
model_pipeline.fit(X_train, y_train)

#  Predict on train and test
print(" Making predictions...")
y_train_pred = model_pipeline.predict(X_train)
y_test_pred = model_pipeline.predict(X_test)

y_train_pred_df = pd.DataFrame(y_train_pred, columns=y_train.columns)
y_test_pred_df = pd.DataFrame(y_test_pred, columns=y_test.columns)

# Evaluate on test
print("\nEvaluation on test set:")
for col in y_test.columns:
    rmse = np.sqrt(mean_squared_error(y_test[col], y_test_pred_df[col]))
    r2 = r2_score(y_test[col], y_test_pred_df[col])
    print(f" {col}: RMSE = {rmse:.2f}, R2 = {r2:.2f}")

# Save model and predictions
pred_dir = os.path.join(base_path, "predictions")
os.makedirs(pred_dir, exist_ok=True)

y_train_pred_df.to_csv(os.path.join(pred_dir, "prediction_train.csv"), index=False)
y_test_pred_df.to_csv(os.path.join(pred_dir, "prediction_test.csv"), index=False)

model_path = os.path.join(base_path, "multioutput_model_pipeline.pkl")
joblib.dump(model_pipeline, model_path)

print(f"\nModel saved to: {model_path}")
print(f"Train predictions saved to: {os.path.join(pred_dir, 'prediction_train.csv')}")
print(f"Test predictions saved to: {os.path.join(pred_dir, 'prediction_test.csv')}")
