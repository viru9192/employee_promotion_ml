import pandas as pd
import joblib
import datetime
import shap
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# -----------------------------
# Load and clean the dataset
# -----------------------------
df = pd.read_csv('../data/hr_dataset.csv')
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(' ', '_')
              .str.replace('?', '', regex=False)
              .str.replace('%', '', regex=False)
              .str.replace('>', '', regex=False)
)

# Fill missing values
df['previous_year_rating'] = df['previous_year_rating'].fillna(df['previous_year_rating'].median())
df['education'] = df['education'].fillna('Unknown')

# Encode categorical features
label_encoders = {}
for col in ['department', 'education', 'gender', 'recruitment_channel']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -----------------------------
# Feature selection
# -----------------------------
features = [
    'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service',
    'avg_training_score', 'awards_won', 'department', 'education',
    'gender', 'recruitment_channel'
]
X = df[features]
y = df['is_promoted']

# -----------------------------
# Train/test split + SMOTE
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# -----------------------------
# Train RandomForest with GridSearchCV
# -----------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [6, 10, None],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1')
rf_grid.fit(X_train_resampled, y_train_resampled)
rf_best_model = rf_grid.best_estimator_

# Evaluation
y_pred_rf_grid = rf_best_model.predict(X_test)
print("\n--- Final Model Evaluation (Random Forest Tuned) ---")
print("Best Params:", rf_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_rf_grid))
print(classification_report(y_test, y_pred_rf_grid))

# -----------------------------
# SHAP Summary Plot
# -----------------------------
if not hasattr(np, 'bool'):
    np.bool = np.bool_

explainer = shap.TreeExplainer(rf_best_model)
sample_X = X_test.sample(n=200, random_state=42)
shap_values = explainer.shap_values(sample_X)

os.makedirs('../app/static', exist_ok=True)
shap.summary_plot(shap_values[1], sample_X, show=False)
plt.savefig("../app/static/shap_summary.png")
plt.close()

# -----------------------------
# Save model bundle (pipeline)
# -----------------------------
os.makedirs('model', exist_ok=True)
model_bundle = {
    'model': rf_best_model,
    'encoders': label_encoders,
    'features': features,
    'version': '2.0.0-tuned-gridsearch',
    'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model_type': 'RandomForestClassifier (Tuned)'
}
joblib.dump(model_bundle, 'model/model.pkl')

print(" Model and SHAP summary saved successfully.")
