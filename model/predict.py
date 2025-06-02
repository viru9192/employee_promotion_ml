# Optional helper script for CLI prediction
import pandas as pd
import pickle

with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    encoders = data['encoders']

sample = {
    'no_of_trainings': 1,
    'age': 35,
    'previous_year_rating': 4.0,
    'length_of_service': 5,
    'avg_training_score': 80,
    'awards_won': 0,
    'kpi_met': 1,
    'department': encoders['department'].transform(['Sales & Marketing'])[0],
    'education': encoders['education'].transform(["Master's & above"])[0],
    'gender': encoders['gender'].transform(['m'])[0],
    'recruitment_channel': encoders['recruitment_channel'].transform(['sourcing'])[0]
}

input_df = pd.DataFrame([sample])
pred = model.predict(input_df)[0]
print('Prediction:', 'Promoted' if pred == 1 else 'Not Promoted')