from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import logging
import sys
import shap
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from datetime import datetime
import json
import traceback

# Configure logging to show in console with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

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

# Load model and verify
try:
    logger.info("Loading model from model/model/model.pkl")
    if not os.path.exists("model/model/model.pkl"):
        raise FileNotFoundError("model.pkl not found in model/model directory")
        
    model_data = joblib.load("model/model/model.pkl")
    logger.info("Model loaded successfully")
    
    # Verify model components
    verify_model_components(model_data)
    logger.info("Model components verified")
    
    model = model_data['model']
    encoders = model_data['encoders']
    features = model_data['features']
    
    # Log model details
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Available features: {features}")
    logger.info(f"Available encoders: {list(encoders.keys())}")
    
    # Test prediction with known good data
    test_data = {
        'no_of_trainings': 1,
        'age': 35,
        'previous_year_rating': 4.0,
        'length_of_service': 5,
        'avg_training_score': 80,
        'awards_won': 0,
        'department': 'Sales & Marketing',
        'education': "Master's & above",
        'gender': 'm',
        'recruitment_channel': 'sourcing'
    }
    
    try:
        # Encode test data
        test_encoded = test_data.copy()
        for col in ['department', 'education', 'gender', 'recruitment_channel']:
            test_encoded[col] = encoders[col].transform([test_data[col]])[0]
            logger.info(f"Test encoding {col}: {test_data[col]} -> {test_encoded[col]}")
        
        test_df = pd.DataFrame([test_encoded])[features]
        test_pred = model.predict(test_df)[0]
        test_proba = model.predict_proba(test_df)[0]
        logger.info(f"Test prediction result: {'Promoted' if test_pred == 1 else 'Not Promoted'}")
        logger.info(f"Test prediction probabilities: {test_proba}")
    except Exception as e:
        logger.error(f"Error in test prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Store predictions in memory (in production, use a database)
predictions_history = []

def generate_shap_plot(model, input_df):
    """Generate SHAP summary plot with error handling"""
    try:
        plt.figure(figsize=(10, 6))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
        
        # Create SHAP summary plot
        shap.summary_plot(shap_values, 
                         input_df, 
                         plot_type="bar",
                         show=False)
        
        # Save plot to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating SHAP plot: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def generate_feature_importance_plot(model, input_df):
    """Generate feature importance bar plot with error handling"""
    try:
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return None
            
        plt.figure(figsize=(10, 6))
        importance = model.feature_importances_
        features = input_df.columns
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        # Create horizontal bar plot
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # Save plot to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating feature importance plot: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def generate_prediction_probability_plot(prediction_proba):
    """Generate prediction probability bar plot with error handling"""
    try:
        plt.figure(figsize=(8, 4))
        labels = ['Not Promoted', 'Promoted']
        colors = ['#ff9999', '#66b3ff']
        
        plt.bar(labels, prediction_proba, color=colors)
        plt.title('Prediction Probabilities')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        
        # Add probability values on top of bars
        for i, v in enumerate(prediction_proba):
            plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
        
        plt.tight_layout()
        
        # Save plot to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating probability plot: {str(e)}")
        logger.error(traceback.format_exc())
        return None

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
                         shap_plot=latest_prediction['shap_plot'],
                         feature_importance_plot=latest_prediction['feature_importance_plot'],
                         probability_plot=latest_prediction['probability_plot'],
                         prediction_data=latest_prediction['input_data'],
                         prediction_result=latest_prediction['result'],
                         prediction_probability=latest_prediction['probability'])

def get_feature_importance(model, input_df):
    """Get feature importance for the prediction"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': input_df.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return feature_importance
    return None

def get_shap_values(model, input_df):
    """Get SHAP values for the prediction"""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        return shap_values[1] if isinstance(shap_values, list) else shap_values
    except Exception as e:
        logger.error(f"Error calculating SHAP values: {str(e)}")
        return None

def create_prediction_explanation(model, input_df, prediction, probability):
    """Create detailed explanation of the prediction"""
    explanation = []
    
    # Get feature importance
    feature_importance = get_feature_importance(model, input_df)
    if feature_importance is not None:
        explanation.append("Top 3 most important features for this prediction:")
        for _, row in feature_importance.head(3).iterrows():
            value = input_df[row['feature']].iloc[0]
            explanation.append(f"- {row['feature']}: {value} (importance: {row['importance']:.3f})")
    
    # Get SHAP values
    shap_values = get_shap_values(model, input_df)
    if shap_values is not None:
        explanation.append("\nFeature contributions to prediction:")
        for feature, value in zip(input_df.columns, shap_values[0]):
            if abs(value) > 0.01:  # Only show significant contributions
                direction = "increases" if value > 0 else "decreases"
                explanation.append(f"- {feature}: {value:.3f} ({direction} promotion probability)")
    
    return "\n".join(explanation)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        logger.info("Received form data:")
        for key, value in data.items():
            logger.info(f"{key}: {value}")

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
        
        logger.info("Processed input dictionary:")
        for key, value in input_dict.items():
            logger.info(f"{key}: {value} (type: {type(value)})")

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
                logger.info(f"Encoded {col}: {original_value} -> {input_dict[col]}")
            except Exception as e:
                logger.error(f"Encoding error for {col}: {input_dict[col]}")
                return render_template("index.html", 
                                    prediction=f"Error: Invalid value for {col}: {input_dict[col]}. "
                                             f"Please select from available options.")

        # Create DataFrame with correct columns
        input_df = pd.DataFrame([input_dict])[features]
        logger.info("Input DataFrame:")
        logger.info(input_df.to_string())
        
        # Get prediction probabilities
        prediction_proba = model.predict_proba(input_df)[0]
        prediction = model.predict(input_df)[0]
        
        logger.info(f"Prediction probabilities: {prediction_proba}")
        logger.info(f"Final prediction: {'Promoted' if prediction == 1 else 'Not Promoted'}")

        # Generate plots
        shap_plot = generate_shap_plot(model, input_df)
        feature_importance_plot = generate_feature_importance_plot(model, input_df)
        probability_plot = generate_prediction_probability_plot(prediction_proba)

        # Get detailed explanation
        explanation = create_prediction_explanation(model, input_df, prediction, prediction_proba)
        logger.info("Prediction explanation:")
        logger.info(explanation)

        result = "Promoted" if prediction == 1 else "Not Promoted"
        probability = f"{prediction_proba[1]*100:.2f}%" if prediction == 1 else f"{prediction_proba[0]*100:.2f}%"
        
        # Store prediction in history
        prediction_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input_data': input_dict,
            'result': result,
            'probability': probability,
            'explanation': explanation,
            'shap_plot': shap_plot,
            'feature_importance_plot': feature_importance_plot,
            'probability_plot': probability_plot
        }
        predictions_history.append(prediction_record)
        
        # Keep only last 10 predictions
        if len(predictions_history) > 10:
            predictions_history.pop(0)

        return render_template("index.html", 
                             prediction=result,
                             probability=probability,
                             explanation=explanation,
                             has_visualizations=True)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template("index.html", 
                             prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
