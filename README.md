# Personality Predictor Web Application

A Flask-based web application that predicts personality types (Introvert/Extrovert) based on social behavior patterns. This application uses a machine learning model trained on social interaction data to make predictions.

## Features

- Simple and intuitive web interface
- Predicts personality type (Introvert/Extrovert)
- Responsive design that works on desktop and mobile devices
- Easy-to-use form for inputting social behavior data

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd personality-predictor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Export the trained model**
   ```bash
   python export_trained_model.py
   ```
   This will create a `personality_model.pkl` file containing the trained model.

## Usage

1. **Start the Flask development server**
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://127.0.0.1:5000
   ```

3. **Fill out the form** with your social behavior information and click "Predict My Personality" to see the result.

## Project Structure

```
personality-predictor/
├── app.py                 # Main Flask application
├── export_model.py        # Script to export the model with sample data
├── export_trained_model.py # Script to train and export the model
├── personality_model.pkl  # Trained model (generated after running export_trained_model.py)
├── requirements.txt       # Python dependencies
├── static/
│   └── css/
│       └── style.css    # Custom CSS styles
└── templates/
    ├── base.html        # Base template with common HTML structure
    └── index.html        # Main page with the prediction form
```

## Dependencies

- Flask==3.0.0
- scikit-learn==1.4.0
- pandas==2.2.0
- numpy==1.26.4
- joblib==1.3.2

## How It Works

1. The application uses a machine learning pipeline that includes:
   - Feature preprocessing (scaling for numerical features, encoding for categorical features)
   - A Decision Tree Classifier for making predictions

2. The model is trained on synthetic data that mimics real-world social behavior patterns related to introversion and extroversion.

3. The web interface collects the following information:
   - Time spent alone (hours/day)
   - Social event attendance (per month)
   - Times going out (per week)
   - Close friends count
   - Social media post frequency
   - Stage fear (Yes/No)
   - Whether socializing is draining (Yes/No)

## Limitations

- The current model is trained on synthetic data and may not be highly accurate for all users
- The application is designed for demonstration purposes
- For production use, the model should be trained on a larger, more diverse dataset

## Future Improvements

- Add user authentication to save prediction history
- Implement more sophisticated machine learning models
- Add data visualization for prediction results
- Include more detailed personality insights
- Improve the UI/UX with animations and better feedback

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Flask and scikit-learn
- Uses Bootstrap for responsive design
