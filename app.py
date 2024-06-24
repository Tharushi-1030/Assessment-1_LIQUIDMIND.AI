from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model
with open('rf_classifier.pkl', 'rb') as f:
    rf_classifier = pickle.load(f)

# Create a Flask app
app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict_segment():
    # Receive input data from the web form
    input_data = request.json

    # Preprocess the input data
    input_df = pd.DataFrame([input_data])
    input_df = input_df.drop(['CUST_ID'], axis=1)  # Assuming 'CUST_ID' is not needed
    input_df.fillna(input_df.mean(), inplace=True)

    # Scale the input features
    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(input_df)

    # Make predictions using the trained model
    predicted_segment = rf_classifier.predict(scaled_input)

    # Map the predicted segment to a meaningful name
    segment_names = {0: 'Segment 1', 1: 'Segment 2', 2: 'Segment 3', 3: 'Segment 4'}
    predicted_segment_name = segment_names[predicted_segment[0]]

    # Return the predicted segment name as JSON response
    return jsonify({'predicted_segment': predicted_segment_name})

if __name__ == '__main__':
    app.run(debug=True)
