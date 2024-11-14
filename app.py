from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Define column names for the input features
column_names = ['Area', 'Production', 'GDP', 'Annual Growth Rate', 'Inflation', 'Rainfall', 'Temperature']

# Initialize and fit the pipeline with training data
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Sample training data for demonstration (replace with actual data for real use)
X_train = np.array([[30, 50, 2.5, 3.0, 1.2, 800, 25], [3500, 16000, 3.0, 3.2, 2.0, 900, 26], [4000, 18000, 3.5, 3.5, 2.5, 1000, 27]])
y_train = np.array([1000, 1050, 1100])

# Fit the pipeline on the training data
X_train_scaled = my_pipeline.fit_transform(X_train)

# Train the Random Forest model
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form as a list of float values
    input_features = [float(x) for x in request.form.values()]
    
    # Convert input features to DataFrame to ensure correct column names
    input_df = pd.DataFrame([input_features], columns=column_names)
    
    # Preprocess the input data with the fitted pipeline
    input_prepared = my_pipeline.transform(input_df)
    
    # Make the prediction
    prediction = model.predict(input_prepared)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Predicted Crop Price: ₹{output}')

if __name__ == "__main__":
    app.run(debug=True)
