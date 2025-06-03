from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import logging
import sys
from datetime import datetime
import json
import traceback

# Configure logging to show in console with more detail
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO to reduce logging
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and related data
model = None
encoders = None
features = None
predictions_history = []

def load_model():
    """Load the model and related data"""
    global model, encoders, features
    
    try:
        logger.info("Starting model loading process...")
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "model", "model.pkl")
        logger.info(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model.pkl not found at {model_path}")
        
        logger.info("Loading model data...")
        model_data = joblib.load(model_path)
        logger.info("Model data loaded successfully")
        
        # Verify model components
        verify_model_components(model_data)
        
        model = model_data['model']
        encoders = model_data['encoders']
        features = model_data['features']
        
        logger.info("Model initialization complete")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def verify_model_components(model_data):
    """Verify all required components are present in the model data"""
    required_components = ['model', 'encoders', 'features']
    missing_components = [comp for comp in required_components if comp not in model_data]
    
    if missing_components:
        raise ValueError(f"Missing required components in model data: {missing_components}")
    
    # Verify encoders
    required_encoders = ['department', 'education', 'gender', 'recruitment_channel']
    missing_encoders = [enc for enc in required_encoders if enc not in model_data['encoders']]
    
    if missing_encoders:
        raise ValueError(f"Missing required encoders: {missing_encoders}")
    
    # Verify features
    if not isinstance(model_data['features'], list):
        raise ValueError("Features must be a list")
    
    return True

# Initialize model on startup
if not load_model():
    logger.error("Failed to load model. Application may not function correctly.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/visualizations')
def visualizations():
    if not predictions_history:
        return render_template('visualizations.html', 
                             has_predictions=False,
                             message="No predictions available. Make a prediction first!")
    
    latest_prediction = predictions_history[-1]
    return render_template('visualizations.html',
                         has_predictions=True,
                         shap_plot=latest_prediction.get('shap_plot'),
                         feature_importance_plot=latest_prediction.get('feature_importance_plot'),
                         probability_plot=latest_prediction.get('probability_plot'),
                         prediction_data=latest_prediction['input_data'],
                         prediction_result=latest_prediction['result'],
                         prediction_probability=latest_prediction['probability'])

def generate_plots(model, input_df, prediction_proba):
    """Generate all plots for visualization"""
    try:
        # Import visualization libraries only when needed
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        import shap
        from io import BytesIO
        import base64
        
        plots = {}
        
        # Generate SHAP plot
        try:
            plt.figure(figsize=(10, 6))
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            buf.seek(0)
            plots['shap_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error generating SHAP plot: {str(e)}")
            plots['shap_plot'] = None
        
        # Generate feature importance plot
        try:
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(10, 6))
                importance = model.feature_importances_
                features = input_df.columns
                
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importance
                }).sort_values('Importance', ascending=True)
                
                plt.barh(importance_df['Feature'], importance_df['Importance'])
                plt.title('Feature Importance')
                plt.xlabel('Importance')
                plt.tight_layout()
                
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                plots['feature_importance_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error generating feature importance plot: {str(e)}")
            plots['feature_importance_plot'] = None
        
        # Generate probability plot
        try:
            plt.figure(figsize=(8, 4))
            labels = ['Not Promoted', 'Promoted']
            colors = ['#ff9999', '#66b3ff']
            
            plt.bar(labels, prediction_proba, color=colors)
            plt.title('Prediction Probabilities')
            plt.ylabel('Probability')
            plt.ylim(0, 1)
            
            for i, v in enumerate(prediction_proba):
                plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
            
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            buf.seek(0)
            plots['probability_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error generating probability plot: {str(e)}")
            plots['probability_plot'] = None
        
        return plots
    except Exception as e:
        logger.error(f"Error in generate_plots: {str(e)}")
        return {}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return render_template("index.html", 
                                 prediction="Error: Model not loaded. Please try again later.")
        
        data = request.form
        
        # Validate required fields
        required_fields = ['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service',
                         'avg_training_score', 'awards_won', 'department', 'education',
                         'gender', 'recruitment_channel']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Collect and organize input with type validation
        try:
            input_dict = {
                'no_of_trainings': int(data['no_of_trainings']),
                'age': int(data['age']),
                'previous_year_rating': float(data['previous_year_rating']),
                'length_of_service': int(data['length_of_service']),
                'avg_training_score': int(data['avg_training_score']),
                'awards_won': int(data['awards_won']),
                'department': data['department'],
                'education': data['education'],
                'gender': data['gender'],
                'recruitment_channel': data['recruitment_channel']
            }
        except ValueError as e:
            raise ValueError(f"Invalid input type: {str(e)}")
        
        # Validate input ranges
        if not (0 <= input_dict['no_of_trainings'] <= 10):
            raise ValueError("Number of trainings must be between 0 and 10")
        if not (18 <= input_dict['age'] <= 65):
            raise ValueError("Age must be between 18 and 65")
        if not (1 <= input_dict['previous_year_rating'] <= 5):
            raise ValueError("Previous year rating must be between 1 and 5")
        if not (0 <= input_dict['length_of_service'] <= 40):
            raise ValueError("Length of service must be between 0 and 40")
        if not (0 <= input_dict['avg_training_score'] <= 100):
            raise ValueError("Average training score must be between 0 and 100")
        if input_dict['awards_won'] not in [0, 1]:
            raise ValueError("Awards won must be 0 or 1")

        # Encode categorical fields
        for col in ['department', 'education', 'gender', 'recruitment_channel']:
            try:
                original_value = input_dict[col]
                input_dict[col] = encoders[col].transform([input_dict[col]])[0]
            except Exception as e:
                logger.error(f"Encoding error for {col}: {input_dict[col]}")
                return render_template("index.html", 
                                    prediction=f"Error: Invalid value for {col}: {input_dict[col]}. "
                                             f"Please select from available options.")

        # Create DataFrame with correct columns
        input_df = pd.DataFrame([input_dict])[features]
        
        # Get prediction probabilities
        prediction_proba = model.predict_proba(input_df)[0]
        prediction = model.predict(input_df)[0]
        
        result = "Promoted" if prediction == 1 else "Not Promoted"
        probability = f"{prediction_proba[1]*100:.2f}%" if prediction == 1 else f"{prediction_proba[0]*100:.2f}%"
        
        # Generate plots only if needed
        plots = generate_plots(model, input_df, prediction_proba)
        
        # Store prediction in history
        prediction_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input_data': input_dict,
            'result': result,
            'probability': probability,
            **plots
        }
        predictions_history.append(prediction_record)
        
        # Keep only last 10 predictions
        if len(predictions_history) > 10:
            predictions_history.pop(0)

        return render_template("index.html", 
                             prediction=result,
                             probability=probability,
                             has_visualizations=True)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template("index.html", 
                             prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
