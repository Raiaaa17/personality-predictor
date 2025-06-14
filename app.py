import os
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-development-only')
app.config['TEMPLATES_AUTO_RELOAD'] = os.environ.get('FLASK_ENV') == 'development'

# Load the trained model
model = joblib.load('personality_model.pkl')

# Feature names in the order expected by the model
FEATURES = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency',
    'Stage_fear',
    'Drained_after_socializing'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Create a DataFrame with the input data and proper column names
        input_data = pd.DataFrame([{
            'Time_spent_Alone': float(data['Time_spent_Alone']),
            'Social_event_attendance': float(data['Social_event_attendance']),
            'Going_outside': float(data['Going_outside']),
            'Friends_circle_size': float(data['Friends_circle_size']),
            'Post_frequency': float(data['Post_frequency']),
            'Stage_fear': data['Stage_fear'],
            'Drained_after_socializing': data['Drained_after_socializing']
        }])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return render_template('index.html', 
                             prediction=prediction,
                             show_result=True,
                             form_data=data)
        
    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_ENV') == 'development')
