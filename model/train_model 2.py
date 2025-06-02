import pandas as pd
import joblib
import datetime
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load and clean dataset
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

# Define features and target
features = [
    'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service',
    'avg_training_score', 'awards_won', 'department', 'education',
    'gender', 'recruitment_channel'
]
X = df[features]
y = df['is_promoted']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# -------------------
# Model 1: Random Forest (Base)
# -------------------
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_model.predict(X_test)
print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# -------------------
# Model 2: XGBoost
# -------------------
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, scale_pos_weight=10, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test)
print("\n--- XGBoost ---")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# -------------------
# Model 3: Random Forest + GridSearchCV
# -------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [6, 10, None],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1')
rf_grid.fit(X_train_resampled, y_train_resampled)
rf_best_model = rf_grid.best_estimator_
y_pred_rf_grid = rf_best_model.predict(X_test)
print("\n--- Random Forest (Tuned) ---")
print("Best Params:", rf_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_rf_grid))
print(classification_report(y_test, y_pred_rf_grid))

# SHAP explanation (for best model)
explainer = shap.TreeExplainer(rf_best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, show=False)
plt.savefig("../app/static/shap_summary.png")

# Save best model
model_metadata = {
    'model': rf_best_model,
    'encoders': label_encoders,
    'version': '2.0.0-tuned-xgboost-shap',
    'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'features': features,
    'model_type': 'RandomForestClassifier (Tuned with GridSearchCV)'
}

joblib.dump(model_metadata, 'model.pkl')