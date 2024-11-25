generate for me a python application that performss machine learning prediction with naive bayes algorithm
it must be an endpoint so the user send the pair attribute and get prediction value for tomorrow as response
generate the code so i can download
here is an example of the dataset

Run pip install flask pandas scikit-learn in your command line to install necessary packages.

Use a tool like Postman or curl to make POST requests to http://127.0.0.1:5000/predict with JSON data similar to:



amount,rate,pair,order_type,created_at,ingest_year,ingest_month,ingest_day,ingest_hour
0.01,"15300000,00",wbtc_jpy,sell,2024-11-23T04:18:27.000Z,2024,11,23,20
0.0,"15300000,00",wbtc_jpy,sell,2024-11-23T03:24:49.000Z,2024,11,23,20
0.01,"15400000,00",wbtc_jpy,sell,2024-11-23T03:23:35.000Z,2024,11,23,20
0.0,"155000000000000,00",wbtc_jpy,buy,2024-11-22T08:21:16.000Z,2024,11,23,20
0.0,"15000000,00",wbtc_jpy,buy,2024-11-21T23:54:39.000Z,2024,11,23,20
0.0,"15000000,00",wbtc_jpy,buy,2024-11-21T23:38:17.000Z,2024,11,23,20

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Sample dataset (for demonstration purposes assume it's already loaded as df)
data = {
    'amount': [0.01, 0.0, 0.01, 0.0, 0.0, 0.0],
    'rate': [15300000.00, 15300000.00, 15400000.00, 15500000.00, 15000000.00, 15000000.00],
    'pair': ['wbtc_jpy', 'wbtc_jpy', 'wbtc_jpy', 'wbtc_jpy', 'wbtc_jpy', 'wbtc_jpy'],
    'order_type': ['sell', 'sell', 'sell', 'buy', 'buy', 'buy'],
    'created_at': ['2024-11-23T04:18:27.000Z', '2024-11-23T03:24:49.000Z', '2024-11-23T03:23:35.000Z',
                   '2024-11-22T08:21:16.000Z', '2024-11-21T23:54:39.000Z', '2024-11-21T23:38:17.000Z'],
    'ingest_year': [2024, 2024, 2024, 2024, 2024, 2024],
    'ingest_month': [11, 11, 11, 11, 11, 11],
    'ingest_day': [23, 23, 23, 22, 21, 21],
    'ingest_hour': [20, 20, 20, 20, 20, 20]
}

df = pd.DataFrame(data)

# Preprocess data
label_encoder = LabelEncoder()
df['order_type'] = label_encoder.fit_transform(df['order_type'])

# Feature and target selection
X = df[['amount', 'rate']]
y = df['order_type']

# Splitting dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    input_data = request.json
    pair = input_data['pair']
    
    # For simplicity, assume 'pair' is constant ('wbtc_jpy'), otherwise, you'd need to encode it.
    amount = float(input_data['amount'])
    rate = float(input_data['rate'])
    
    # Prepare input data for prediction
    input_features = np.array([[amount, rate]])
    
    # Make prediction
    prediction = model.predict(input_features)
    
    # Decode the prediction
    prediction_decoded = label_encoder.inverse_transform(prediction)[0]
    
    # Return the prediction result
    return jsonify({
        'pair': pair,
        'predicted_order_type': prediction_decoded
    })

if __name__ == '__main__':
    app.run(debug=True)