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

# Create the final pipeline with the same parameters as in the notebook
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42))
])

# Create sample training data that matches the notebook's training data structure
# This is a simplified version - in a real scenario, you would load the actual training data
X_train = pd.DataFrame({
    'Time_spent_Alone': [5, 8, 3, 2, 7, 4, 6, 1, 9, 5],
    'Social_event_attendance': [3, 1, 5, 4, 2, 5, 3, 2, 1, 4],
    'Going_outside': [2, 1, 3, 4, 2, 3, 2, 1, 4, 2],
    'Friends_circle_size': [10, 5, 15, 20, 8, 12, 7, 25, 3, 18],
    'Post_frequency': [2, 1, 5, 3, 2, 4, 1, 6, 2, 3],
    'Stage_fear': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
    'Drained_after_socializing': ['No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
})

y_train = np.array(['Introvert', 'Extrovert', 'Extrovert', 'Introvert', 'Extrovert', 
                   'Introvert', 'Extrovert', 'Introvert', 'Extrovert', 'Introvert'])

# Fit the model with the training data
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'personality_model.pkl')
print("Trained model exported to 'personality_model.pkl'")
print("Model accuracy on training data:", model.score(X_train, y_train))
