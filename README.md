# Employee Promotion Prediction System

A machine learning-based web application that predicts employee promotion likelihood and provides detailed explanations of the predictions using SHAP values and feature importance analysis.

## 🌟 Features

- **Interactive Prediction Interface**: User-friendly form to input employee details
- **Real-time Predictions**: Instant promotion probability predictions
- **Detailed Explanations**: 
  - SHAP value analysis for each prediction
  - Feature importance visualization
  - Prediction probability breakdown
- **Modern UI/UX**:
  - Responsive design
  - Clean and intuitive interface
  - Interactive visualizations
  - Smooth animations and transitions

## 🛠️ Technology Stack

- **Backend**:
  - Python 3.10+
  - Flask (Web Framework)
  - XGBoost (Machine Learning Model)
  - SHAP (Model Explanation)
  - scikit-learn (Data Processing)
  - pandas & numpy (Data Manipulation)
  - matplotlib & seaborn (Visualization)

- **Frontend**:
  - HTML5
  - CSS3
  - Bootstrap 5.3.2
  - Bootstrap Icons
  - Google Fonts (Inter)

## 📋 Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd employee_promotion_ml
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the model file is present:
   - Place your trained model file (`model.pkl`) in the `model/` directory
   - The model file should contain:
     - Trained XGBoost model
     - Feature encoders
     - Feature list

## 💻 Usage

1. Start the Flask application:
   ```bash
   python app/app.py
   ```

2. Access the application:
   - Open your web browser
   - Navigate to `http://127.0.0.1:5000`

3. Making Predictions:
   - Fill out the employee details form
   - Click "Predict" to get the promotion prediction
   - View detailed explanations and visualizations

4. Viewing Visualizations:
   - Click "Visualizations" in the navigation bar
   - Explore SHAP summary plots
   - Review feature importance
   - Analyze prediction probabilities

## 📊 Input Features

The model considers the following employee attributes:

1. **Numerical Features**:
   - Number of Trainings (0-10)
   - Age (18-65)
   - Previous Year Rating (1-5)
   - Length of Service (0-40)
   - Average Training Score (0-100)
   - Awards Won (0 or 1)

2. **Categorical Features**:
   - Department (Sales & Marketing, Operations, Technology, etc.)
   - Education (Bachelor's, Master's & above, Below Secondary)
   - Gender (Male, Female, Other)
   - Recruitment Channel (Sourcing, Referred, Other)

## 📈 Model Performance

The application provides three types of visualizations:

1. **SHAP Summary Plot**:
   - Shows how each feature contributes to the prediction
   - Dynamic for each prediction
   - Helps understand feature impact

2. **Feature Importance Plot**:
   - Displays global feature importance
   - Static across all predictions
   - Shows overall model behavior

3. **Prediction Probability Plot**:
   - Visualizes the confidence of predictions
   - Shows probability distribution
   - Helps in decision-making

## 🔍 Understanding the Results

1. **Prediction Result**:
   - "Promoted" or "Not Promoted"
   - Confidence percentage
   - Detailed explanation of factors

2. **Visualizations**:
   - SHAP values show feature contributions
   - Feature importance shows overall model behavior
   - Probability plot shows prediction confidence

## 🛠️ Development

### Project Structure
```
employee_promotion_ml/
├── app/
│   ├── app.py              # Flask application
│   ├── templates/          # HTML templates
│   │   ├── index.html      # Main prediction page
│   │   └── visualizations.html  # Visualizations page
│   └── static/            # Static files (if any)
├── model/
│   └── model.pkl          # Trained model file
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

### Adding New Features

1. **Model Updates**:
   - Retrain the model with new data
   - Update the model file in the `model/` directory
   - Ensure feature encoders are compatible

2. **UI Modifications**:
   - Edit HTML templates in `app/templates/`
   - Modify CSS styles in the template files
   - Update JavaScript if needed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Viranchi More

## 🙏 Acknowledgments

- XGBoost team for the machine learning framework
- SHAP team for the model explanation library
- Flask team for the web framework
- Bootstrap team for the frontend framework

## 📞 Support

For support, please:
1. Check the documentation
2. Open an issue in the repository
3. Contact the maintainers

## 🔄 Updates

- Latest update: Added modern UI with responsive design
- Added detailed visualizations
- Improved prediction explanations
- Enhanced error handling

---
