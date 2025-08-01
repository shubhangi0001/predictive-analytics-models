import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

data_path = "D:/employee_details/dashboard2/employee_data_final_processed.csv"
pred_dir = "D:/employee_details/dashboard2/predictions"
os.makedirs(pred_dir, exist_ok=True)

df = pd.read_csv(data_path)

target_cols = ['Performance Score', 'engagementsurvey', 'tenure_years']
id_cols = ['Employee_Name', 'EmpID']
X = df.drop(columns=target_cols)
y = df[target_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train.to_csv(f"{pred_dir}/y_train.csv", index=False)
y_test.to_csv(f"{pred_dir}/y_test.csv", index=False)

categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', MultiOutputRegressor(RandomForestRegressor(random_state=42)))
])

print("🔁 Training model with preprocessing...")
model.fit(X_train, y_train)

print("Making predictions...")
y_pred_train = pd.DataFrame(model.predict(X_train), columns=target_cols)
y_pred_test = pd.DataFrame(model.predict(X_test), columns=target_cols)

train_results = pd.concat([X_train[id_cols].reset_index(drop=True), y_train.reset_index(drop=True), y_pred_train], axis=1)
test_results = pd.concat([X_test[id_cols].reset_index(drop=True), y_test.reset_index(drop=True), y_pred_test], axis=1)
all_results = pd.concat([train_results, test_results], axis=0)

train_results.to_csv(f"{pred_dir}/prediction_train.csv", index=False)
test_results.to_csv(f"{pred_dir}/prediction_test.csv", index=False)
all_results.to_csv(f"{pred_dir}/prediction_all.csv", index=False)

print("\nEvaluation:")
for col in target_cols:
    rmse = mean_squared_error(y_test[col], y_pred_test[col], squared=False)
    r2 = r2_score(y_test[col], y_pred_test[col])
    print(f" {col}: RMSE = {rmse:.2f}, R2 = {r2:.2f}")

print("\n Model training, prediction, and evaluation completed.")
