import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

# Define the features
numeric_features = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]

categorical_features = [
    'Stage_fear',
    'Drained_after_socializing'
]

# Create transformers
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder())
])

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the final pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42))
])

# Create sample training data to fit the pipeline
# This is a simplified version of the training data structure
# In a real scenario, you would load your actual training data
X_sample = pd.DataFrame({
    'Time_spent_Alone': [5, 8, 3, 2, 7],
    'Social_event_attendance': [3, 1, 5, 4, 2],
    'Going_outside': [2, 1, 3, 4, 2],
    'Friends_circle_size': [10, 5, 15, 20, 8],
    'Post_frequency': [2, 1, 5, 3, 2],
    'Stage_fear': ['Yes', 'No', 'No', 'Yes', 'No'],
    'Drained_after_socializing': ['No', 'Yes', 'No', 'No', 'Yes']
})

y_sample = np.array(['Introvert', 'Extrovert', 'Extrovert', 'Introvert', 'Extrovert'])

# Fit the model with sample data
model.fit(X_sample, y_sample)

# Save the model
joblib.dump(model, 'personality_model.pkl')
print("Model trained with sample data and saved as 'personality_model.pkl'")
