from flask import Flask, jsonify
import pickle
import pandas as pd
import xgboost as xgb

# Initialize Flask app
app = Flask(__name__)

# Load the trained XGBoost model
model_filename = 'xgboost_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Example route to predict with the loaded model
@app.route('/')
def predict():
    # Example data (you should replace this with actual data processing)
    example_data = {
        'year': 2023,
        'month': 6,
        'day': 28,
        'dayofweek': 2,
        'price_lag_1': 50,
        'price_lag_7': 48,
        'price_rolling_mean_7': 52,
        'price_rolling_mean_30': 55,
        'mrp_lag_1': 60,
        'mrp_lag_7': 58,
        'mrp_rolling_mean_7': 62,
        'mrp_rolling_mean_30': 65,
        'rating_lag_1': 4.5,
        'rating_lag_7': 4.6,
        'rating_rolling_mean_7': 4.4,
        'rating_rolling_mean_30': 4.3,
        'ratingTotal_lag_1': 100,
        'ratingTotal_lag_7': 105,
        'ratingTotal_rolling_mean_7': 98,
        'ratingTotal_rolling_mean_30': 95,
        'discount_lag_1': 0.1,
        'discount_lag_7': 0.15,
        'discount_rolling_mean_7': 0.12,
        'discount_rolling_mean_30': 0.1
    }
    
    # Convert example_data to a DataFrame
    example_df = pd.DataFrame([example_data])

    # Make prediction
    prediction = model.predict(example_df)
    
    # Return prediction as JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=False)
